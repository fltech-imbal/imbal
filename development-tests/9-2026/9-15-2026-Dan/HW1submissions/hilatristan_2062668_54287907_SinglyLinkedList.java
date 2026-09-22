import java.util.Iterator;
import java.util.NoSuchElementException;

public class SinglyLinkedList<E> implements Iterable<E> {
    private static class Node<E> {
        E element;
        Node<E> next;

        Node(E element, Node<E> next) {
            this.element = element;
            this.next = next;
        }
    }

    private Node<E> head;
    private Node<E> tail;
    private int size;

    public int size() { return size; }
    public boolean isEmpty() { return size == 0; }
    public E first() { return isEmpty() ? null : head.element; }

    public void addFirst(E element) {
        head = new Node<E>(element, head);
        if (tail == null) tail = head;
        size++;
    }

    public void addLast(E element) {
        Node<E> newest = new Node<E>(element, null);
        if (tail == null) head = newest;
        else tail.next = newest;
        tail = newest;
        size++;
    }

    public E removeFirst() {
        if (isEmpty()) return null;
        E answer = head.element;
        head = head.next;
        size--;
        if (head == null) tail = null;
        return answer;
    }

    /** Removes the first node containing the same object reference. */
    public boolean remove(E element) {
        Node<E> previous = null;
        Node<E> current = head;
        while (current != null) {
            if (current.element == element) {
                if (previous == null) head = current.next;
                else previous.next = current.next;
                if (current == tail) tail = previous;
                size--;
                return true;
            }
            previous = current;
            current = current.next;
        }
        return false;
    }

    public Iterator<E> iterator() {
        return new Iterator<E>() {
            private Node<E> current = head;
            public boolean hasNext() { return current != null; }
            public E next() {
                if (current == null) throw new NoSuchElementException();
                E answer = current.element;
                current = current.next;
                return answer;
            }
            public void remove() { throw new UnsupportedOperationException(); }
        };
    }
}
