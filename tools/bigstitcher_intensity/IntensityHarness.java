import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;

import mpicbg.models.*;
import mpicbg.spim.data.sequence.ViewId;
import net.imglib2.*;
import net.imglib2.RandomAccess;
import mpicbg.models.Point;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import net.imglib2.view.Views;
import net.preibisch.mvrecon.process.boundingbox.BoundingBoxMaximal;
import net.preibisch.mvrecon.process.fusion.intensityadjust.IntensityAdjustmentTools;
import net.preibisch.mvrecon.process.fusion.transformed.TransformView;
import net.preibisch.mvrecon.process.fusion.transformed.TransformVirtual;

/**
 * Runs BigStitcher's intensity adjustment (multiview-reconstruction 9.1.1,
 * IntensityAdjustmentTools.computeIntensityAdjustment) on views given as raw
 * float32 arrays + view->world affines, so a Python port can be checked
 * against it on identical input.
 *
 * REAL BigStitcher classes: BoundingBoxMaximal (bounding box), TransformVirtual
 * (scaling), TransformView/TransformedInputRandomAccessible (linear resampling,
 * -1 outside), IntensityAdjustmentTools.runGlobal (solver + min-offset shift),
 * mpicbg 1.6.6 models/TileConfiguration, imglib2 ValuePair, java.util.Random.
 *
 * COPIED VERBATIM (it is fused to SpimData/ImgLoader upstream): the loop that
 * collects intensity PointMatches over the bounding box and the
 * Random(344) thinning to < maxMatches.
 *
 * Usage: java IntensityHarness case.txt out.txt [matchesDumpDir]
 */
public class IntensityHarness
{
	public static void main( final String[] args ) throws Exception
	{
		final List< String > lines = Files.readAllLines( Paths.get( args[ 0 ] ) );
		final Map< String, String > kv = new HashMap<>();
		final List< String[] > viewLines = new ArrayList<>();
		for ( final String l : lines )
		{
			final String t = l.trim();
			if ( t.isEmpty() || t.startsWith( "#" ) ) continue;
			final String[] p = t.split( "\\s+" );
			if ( p[ 0 ].equals( "view" ) ) viewLines.add( p );
			else kv.put( p[ 0 ], p[ 1 ] );
		}
		double downsampling = Double.parseDouble( kv.get( "downsampling" ) );
		final int maxMatches = Integer.parseInt( kv.get( "maxMatches" ) );
		final boolean affine = Integer.parseInt( kv.get( "affine" ) ) != 0;
		final double regTrans = Double.parseDouble( kv.get( "regTrans" ) );
		final double regIdentity = Double.parseDouble( kv.get( "regIdentity" ) );
		final int solverIterations = Integer.parseInt( kv.getOrDefault( "solverIterations", "0" ) );
		if ( downsampling == 1.0 ) downsampling = Double.NaN; // as Intensity_Adjustment

		// --- views ---------------------------------------------------------
		final List< ViewId > viewIds = new ArrayList<>();
		final HashMap< ViewId, Dimensions > dims = new HashMap<>();
		final HashMap< ViewId, AffineTransform3D > regs = new HashMap<>();
		final List< ArrayImg< FloatType, FloatArray > > imgs = new ArrayList<>();
		for ( final String[] p : viewLines )
		{
			// view <id> <dx> <dy> <dz> <file> m00 m01 m02 m03 m10 ... m23
			final int id = Integer.parseInt( p[ 1 ] );
			final long dx = Long.parseLong( p[ 2 ] ), dy = Long.parseLong( p[ 3 ] ), dz = Long.parseLong( p[ 4 ] );
			final float[] data = readFloats( p[ 5 ], (int)( dx * dy * dz ) );
			final ArrayImg< FloatType, FloatArray > img = ArrayImgs.floats( data, dx, dy, dz );
			final AffineTransform3D t = new AffineTransform3D();
			final double[] m = new double[ 12 ];
			for ( int i = 0; i < 12; ++i ) m[ i ] = Double.parseDouble( p[ 6 + i ] );
			t.set( m );
			final ViewId v = new ViewId( 0, id );
			viewIds.add( v );
			dims.put( v, new FinalDimensions( dx, dy, dz ) );
			regs.put( v, t );
			imgs.add( img );
		}

		// --- bounding box: "All Views" maximal box (real BigStitcher code) ---
		final double[] minBB = new double[ 3 ], maxBB = new double[ 3 ];
		BoundingBoxMaximal.computeMaxBoundingBoxDimensions( viewIds, dims, regs, minBB, maxBB );
		Interval bb = new FinalInterval(
				toLong( BoundingBoxMaximal.approximateLowerBound( minBB ) ),
				toLong( BoundingBoxMaximal.approximateUpperBound( maxBB ) ) );
		final PrintWriter out = new PrintWriter( new FileWriter( args[ 1 ] ) );
		out.println( "bb " + str( bb ) );

		// --- from here: computeIntensityAdjustment, with the ImgLoader swapped
		//     for the arrays above. Lines marked // VERBATIM are copied as-is.
		if ( !Double.isNaN( downsampling ) )                                   // VERBATIM
			bb = TransformVirtual.scaleBoundingBox( bb, 1.0 / downsampling );    // VERBATIM
		out.println( "scaledbb " + str( bb ) );

		final ArrayList< RandomAccessibleInterval< FloatType > > images = new ArrayList<>();
		for ( int vi = 0; vi < viewIds.size(); ++vi )
		{
			AffineTransform3D model = regs.get( viewIds.get( vi ) );
			if ( !Double.isNaN( downsampling ) )                               // VERBATIM
			{
				model = model.copy();                                          // VERBATIM
				TransformVirtual.scaleTransform( model, 1.0 / downsampling );  // VERBATIM
			}
			// upstream: DownsampleTools.openDownsampled picks a pyramid level and
			// adjusts `model`; here the array IS the level (no pyramid).
			images.add( TransformView.transformView( imgs.get( vi ), model, bb, -1, 1 ) ); // VERBATIM
		}

		final int m = images.size();                                                                   // VERBATIM
		final HashMap< Integer, ViewId > viewMap = new HashMap<>();                                    // VERBATIM
		final Cursor< FloatType > cursor = Views.iterable( images.get( 0 ) ).localizingCursor();       // VERBATIM
		final ArrayList< RandomAccess< FloatType > > accesses = new ArrayList<>();                     // VERBATIM
		for ( int i = 0; i < m; ++i )                                                                  // VERBATIM
		{
			accesses.add( images.get( i ).randomAccess() );                                            // VERBATIM
			viewMap.put( i, viewIds.get( i ) );                                                        // VERBATIM
		}
		final HashMap< Pair< Integer, Integer >, ArrayList< PointMatch > > intensityMatches = new HashMap<>(); // VERBATIM
		for ( int i = 0; i < m - 1; ++i )                                                              // VERBATIM
			for ( int j = i + 1; j < m; ++j )                                                          // VERBATIM
				intensityMatches.put( new ValuePair< Integer, Integer >( i, j ), new ArrayList<>() );  // VERBATIM
		final ArrayList< Pair< Integer, Float > > values = new ArrayList<>();                          // VERBATIM
		while ( cursor.hasNext() )                                                                     // VERBATIM
		{
			cursor.fwd();                                                                              // VERBATIM
			values.clear();                                                                            // VERBATIM
			for ( int i = 0; i < m; ++i )                                                              // VERBATIM
			{
				final RandomAccess< FloatType > r = accesses.get( i );                                 // VERBATIM
				r.setPosition( cursor );                                                               // VERBATIM
				final float value = r.get().get();                                                     // VERBATIM
				if ( value >= 0 )                                                                      // VERBATIM
					values.add( new ValuePair< Integer, Float >( i, value ) );                         // VERBATIM
			}
			if ( values.size() > 1 )                                                                   // VERBATIM
				for ( int i = 0; i < values.size() - 1; ++i )                                          // VERBATIM
					for ( int j = i + 1; j < values.size(); ++j )                                      // VERBATIM
					{
						final ArrayList< PointMatch > matches = intensityMatches.get( new ValuePair< Integer, Integer >( values.get( i ).getA(), values.get( j ).getA() ) ); // VERBATIM
						matches.add( new PointMatch( new Point( new double[] { values.get( i ).getB() } ), new Point( new double[] { values.get( j ).getB() } ) ) );          // VERBATIM
					}
		}
		// record the pre-thinning counts in iteration order of the map
		final List< String > order = new ArrayList<>();
		final Map< String, Integer > before = new HashMap<>();
		for ( final Map.Entry< Pair< Integer, Integer >, ArrayList< PointMatch > > e : intensityMatches.entrySet() )
		{
			final String k = e.getKey().getA() + " " + e.getKey().getB();
			order.add( k );
			before.put( k, e.getValue().size() );
		}
		final Random rnd = new Random( 344 );                                                          // VERBATIM
		for ( final ArrayList< PointMatch > matches : intensityMatches.values() )                      // VERBATIM
		{
			while ( matches.size() >= maxMatches )                                                     // VERBATIM
				matches.remove( rnd.nextInt( matches.size() ) );                                       // VERBATIM
		}
		out.println( "maporder " + String.join( ",", order ) );
		for ( final Map.Entry< Pair< Integer, Integer >, ArrayList< PointMatch > > e : intensityMatches.entrySet() )
		{
			final String k = e.getKey().getA() + " " + e.getKey().getB();
			double s1 = 0, s2 = 0, s11 = 0, s12 = 0;
			for ( final PointMatch pm : e.getValue() )
			{
				final double a = pm.getP1().getL()[ 0 ], b = pm.getP2().getL()[ 0 ];
				s1 += a; s2 += b; s11 += a * a; s12 += a * b;
			}
			out.println( "pair " + k + " " + before.get( k ) + " " + e.getValue().size() + " " + s1 + " " + s2 + " " + s11 + " " + s12 );
			if ( args.length > 2 && e.getValue().size() > 0 )
			{
				try ( DataOutputStream d = new DataOutputStream( new BufferedOutputStream(
						new FileOutputStream( args[ 2 ] + "/pair_" + e.getKey().getA() + "_" + e.getKey().getB() + ".f64" ) ) ) )
				{
					for ( final PointMatch pm : e.getValue() )
					{
						d.writeDouble( pm.getP1().getL()[ 0 ] );
						d.writeDouble( pm.getP2().getL()[ 0 ] );
					}
				}
			}
		}

		// --- solver: exactly as Intensity_Adjustment builds the model -------
		final HashMap< ViewId, AffineModel1D > result;
		if ( affine )
		{
			final InterpolatedAffineModel1D< InterpolatedAffineModel1D< AffineModel1D, TranslationModel1D >, IdentityModel > model =
					new InterpolatedAffineModel1D<>(
							new InterpolatedAffineModel1D<>( new AffineModel1D(), new TranslationModel1D(), regTrans ),
							new IdentityModel(), regIdentity );
			result = solverIterations > 0 ? runGlobalLong( intensityMatches, viewMap, model, solverIterations )
					: IntensityAdjustmentTools.runGlobal( intensityMatches, viewMap, model );
		}
		else
		{
			final InterpolatedAffineModel1D< TranslationModel1D, IdentityModel > model =
					new InterpolatedAffineModel1D<>( new TranslationModel1D(), new IdentityModel(), regIdentity );
			result = solverIterations > 0 ? runGlobalLong( intensityMatches, viewMap, model, solverIterations )
					: IntensityAdjustmentTools.runGlobal( intensityMatches, viewMap, model );
		}
		final double[] arr = new double[ 2 ];
		for ( int i = 0; i < m; ++i )
		{
			result.get( viewMap.get( i ) ).toArray( arr );
			out.println( "model " + i + " " + arr[ 0 ] + " " + arr[ 1 ] );
		}
		out.close();
	}

	/**
	 * runGlobal with only the iteration budget changed: to find the fixed point
	 * that 300 shuffled sweeps approach. Everything else copied from
	 * IntensityAdjustmentTools.runGlobal (9.1.1).
	 */
	static < M extends Model< M > & Affine1D< M > > HashMap< ViewId, AffineModel1D > runGlobalLong(
			final HashMap< Pair< Integer, Integer >, ArrayList< PointMatch > > intensityMatches,
			final HashMap< Integer, ViewId > viewMap, final M model, final int iterations ) throws Exception
	{
		final int m = viewMap.keySet().size();
		final TileConfiguration tc = new TileConfiguration();
		final HashMap< Integer, Tile< M > > tiles = new HashMap<>();
		for ( int i = 0; i < m; ++i ) tiles.put( i, new Tile<>( model.copy() ) );
		for ( int i = 0; i < m - 1; ++i )
			for ( int j = i + 1; j < m; ++j )
			{
				final ArrayList< PointMatch > c = intensityMatches.get( new ValuePair< Integer, Integer >( i, j ) );
				if ( c.size() > 0 )
				{
					final ArrayList< PointMatch > pm = new ArrayList<>( c );
					tiles.get( i ).addMatches( pm );
					tiles.get( j ).addMatches( PointMatch.flip( pm ) );
					tiles.get( i ).addConnectedTile( tiles.get( j ) );
					tiles.get( j ).addConnectedTile( tiles.get( i ) );
				}
			}
		for ( int i = 0; i < m; ++i )
			if ( tiles.get( i ).getConnectedTiles().size() > 0 ) tc.addTile( tiles.get( i ) );
		tc.preAlign();
		tc.optimize( 0.0, iterations, iterations );
		final HashMap< ViewId, AffineModel1D > result = new HashMap<>();
		final double[] a = new double[ 2 ];
		double minOffset = Double.MAX_VALUE;
		for ( int i = 0; i < m; ++i ) { tiles.get( i ).getModel().toArray( a ); minOffset = Math.min( minOffset, a[ 1 ] ); }
		for ( int i = 0; i < m; ++i )
		{
			tiles.get( i ).getModel().toArray( a );
			a[ 1 ] -= minOffset;
			final AffineModel1D mv = new AffineModel1D();
			mv.set( a[ 0 ], a[ 1 ] );
			result.put( viewMap.get( i ), mv );
		}
		return result;
	}

	static float[] readFloats( final String path, final int n ) throws IOException
	{
		final ByteBuffer b = ByteBuffer.allocate( n * 4 ).order( ByteOrder.LITTLE_ENDIAN );
		try ( FileChannel c = FileChannel.open( Paths.get( path ) ) ) { while ( b.hasRemaining() && c.read( b ) >= 0 ) {} }
		b.flip();
		final float[] f = new float[ n ];
		b.asFloatBuffer().get( f );
		return f;
	}

	static long[] toLong( final int[] a ) { final long[] l = new long[ a.length ]; for ( int i = 0; i < a.length; ++i ) l[ i ] = a[ i ]; return l; }

	static String str( final Interval i )
	{
		final StringBuilder s = new StringBuilder();
		for ( int d = 0; d < i.numDimensions(); ++d ) s.append( i.min( d ) ).append( ' ' );
		for ( int d = 0; d < i.numDimensions(); ++d ) s.append( i.max( d ) ).append( ' ' );
		return s.toString().trim();
	}
}
