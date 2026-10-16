import java.util.*;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
/** Ground truth for the Python port's java.util.Random / HashMap / ArrayList pieces. */
public class JavaBits {
  public static void main(String[] a) {
    StringBuilder sb = new StringBuilder("{\n");
    // Random(344).nextInt over a spread of bounds, incl. powers of two and near 2^31
    Random r = new Random(344);
    int[] bounds = {100, 64, 1, 7, 1000000, 1 << 20, 2147483647, 1073741825, 3, 12345};
    sb.append("\"random_bounds\": ").append(Arrays.toString(bounds)).append(",\n\"random_344\": [");
    for (int k = 0; k < 200; k++) { sb.append(r.nextInt(bounds[k % bounds.length])); if (k < 199) sb.append(','); }
    sb.append("],\n");
    // HashMap<ValuePair> iteration order for several view counts
    for (int m : new int[]{9, 20, 42, 70}) {
      HashMap<Pair<Integer,Integer>, Integer> h = new HashMap<>();
      for (int i = 0; i < m - 1; i++) for (int j = i + 1; j < m; j++) h.put(new ValuePair<>(i, j), 0);
      sb.append("\"order_").append(m).append("\": [");
      boolean first = true;
      for (Pair<Integer,Integer> k : h.keySet()) { if (!first) sb.append(','); first = false; sb.append('[').append(k.getA()).append(',').append(k.getB()).append(']'); }
      sb.append("],\n");
    }
    // ArrayList thinning: survivors (original indices) for one shared Random(344) over three lists
    Random rt = new Random(344);
    int[][] cases = {{5000, 300}, {299, 300}, {300, 300}, {20000, 1000}};
    sb.append("\"thin_cases\": [");
    for (int c = 0; c < cases.length; c++) {
      ArrayList<Integer> l = new ArrayList<>();
      for (int i = 0; i < cases[c][0]; i++) l.add(i);
      while (l.size() >= cases[c][1]) l.remove(rt.nextInt(l.size()));
      sb.append("{\"n\": ").append(cases[c][0]).append(", \"max\": ").append(cases[c][1]).append(", \"survivors\": ").append(l.toString()).append("}");
      if (c < cases.length - 1) sb.append(',');
    }
    sb.append("]\n}\n");
    System.out.print(sb);
  }
}
