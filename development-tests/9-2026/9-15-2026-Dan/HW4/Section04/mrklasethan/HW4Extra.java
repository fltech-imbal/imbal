import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

// builds the heap starting from the bottom
public class HW4Extra {
    public static void initialize(HeapPriorityQueue<Double, String> queue,
                                  String[] names, double[] distances, int numEntries) {
        if (numEntries < 0 || numEntries > names.length || numEntries > distances.length)
            throw new IllegalArgumentException("Invalid number of entries");
        queue.heap.clear();
        for (int i = 0; i < numEntries; i++)
            queue.heap.add(new AbstractPriorityQueue.PQEntry<>(distances[i], names[i]));
        // moves each parent down to fix the heap
        for (int i = numEntries / 2 - 1; i >= 0; i--) queue.downheap(i);
    }

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length != 1) {
            System.err.println("Usage: java HW4Extra ratings.txt");
            return;
        }
        try (Scanner file = new Scanner(new File(args[0]))) {
            String target = file.next();
            String[] names = new String[100];
            int[][] ratings = new int[100][10];
            int n = 0, targetIndex = -1;
            while (file.hasNext()) {
                names[n] = file.next();
                if (names[n].equals(target)) targetIndex = n;
                for (int j = 0; j < 10; j++) ratings[n][j] = file.nextInt();
                n++;
            }
            if (targetIndex < 0) throw new IllegalArgumentException("Target not found");
            String[] otherNames = new String[n - 1];
            double[] distances = new double[n - 1];
            int k = 0;
            for (int i = 0; i < n; i++) {
                if (i == targetIndex) continue;
                int common = 0, sum = 0;
                for (int j = 0; j < 10; j++) {
                    if (ratings[i][j] != 0 && ratings[targetIndex][j] != 0) {
                        common++;
                        sum += Math.abs(ratings[i][j] - ratings[targetIndex][j]);
                    }
                }
                otherNames[k] = names[i];
                distances[k] = common == 0 ? Double.POSITIVE_INFINITY :
                    (double) sum / common + 1.0 / common;
                k++;
            }
            HeapPriorityQueue<Double, String> queue = new HeapPriorityQueue<>();
            initialize(queue, otherNames, distances, k);
            while (!queue.isEmpty()) {
                Entry<Double, String> entry = queue.removeMin();
                System.out.println(entry.getValue() + " " + entry.getKey());
            }
        }
    }
}
