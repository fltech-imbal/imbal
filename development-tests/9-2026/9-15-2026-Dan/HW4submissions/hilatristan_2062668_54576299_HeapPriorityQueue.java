import java.util.ArrayList;
import java.util.Comparator;

/** An implementation of a priority queue using an array-based heap. */
public class HeapPriorityQueue<K,V> extends AbstractPriorityQueue<K,V> {
    /** Primary collection of priority-queue entries. */
    protected ArrayList<Entry<K,V>> heap = new ArrayList<>();

    /** Creates an empty priority queue using natural key order. */
    public HeapPriorityQueue() { super(); }

    /** Creates an empty priority queue using the supplied comparator. */
    public HeapPriorityQueue(Comparator<K> comp) { super(comp); }

    protected int parent(int index) { return (index - 1) / 2; }
    protected int left(int index) { return 2 * index + 1; }
    protected int right(int index) { return 2 * index + 2; }
    protected boolean hasLeft(int index) { return left(index) < heap.size(); }
    protected boolean hasRight(int index) { return right(index) < heap.size(); }

    /** Exchanges two heap entries. */
    protected void swap(int first, int second) {
        Entry<K,V> temporary = heap.get(first);
        heap.set(first, heap.get(second));
        heap.set(second, temporary);
    }

    /** Moves an entry upward until heap order is restored. */
    protected void upheap(int index) {
        while (index > 0) {
            int parentIndex = parent(index);
            if (compare(heap.get(index), heap.get(parentIndex)) >= 0) {
                break;
            }
            swap(index, parentIndex);
            index = parentIndex;
        }
    }

    /** Moves an entry downward until heap order is restored. */
    protected void downheap(int index) {
        while (hasLeft(index)) {
            int smallerChild = left(index);
            if (hasRight(index) && compare(heap.get(left(index)), heap.get(right(index))) > 0) {
                smallerChild = right(index);
            }
            if (compare(heap.get(smallerChild), heap.get(index)) >= 0) {
                break;
            }
            swap(index, smallerChild);
            index = smallerChild;
        }
    }

    /**
     * Initializes this queue with bottom-up heap construction.
     * The supplied entries are copied, so later array changes do not affect the heap.
     */
    public void initialize(Entry<K,V>[] entryArray, int numEntries) {
        if (entryArray == null || numEntries < 0 || numEntries > entryArray.length) {
            throw new IllegalArgumentException("invalid entry array or entry count");
        }

        heap.clear();
        for (int index = 0; index < numEntries; index++) {
            if (entryArray[index] == null) {
                throw new IllegalArgumentException("entry array contains null");
            }
            checkKey(entryArray[index].getKey());
            heap.add(entryArray[index]);
        }
        for (int index = parent(heap.size() - 1); index >= 0; index--) {
            downheap(index);
        }
    }

    @Override
    public int size() { return heap.size(); }

    @Override
    public Entry<K,V> min() {
        return heap.isEmpty() ? null : heap.get(0);
    }

    @Override
    public Entry<K,V> insert(K key, V value) {
        checkKey(key);
        Entry<K,V> newest = new PQEntry<>(key, value);
        heap.add(newest);
        upheap(heap.size() - 1);
        return newest;
    }

    @Override
    public Entry<K,V> removeMin() {
        if (heap.isEmpty()) {
            return null;
        }
        Entry<K,V> answer = heap.get(0);
        swap(0, heap.size() - 1);
        heap.remove(heap.size() - 1);
        if (!heap.isEmpty()) {
            downheap(0);
        }
        return answer;
    }
}
