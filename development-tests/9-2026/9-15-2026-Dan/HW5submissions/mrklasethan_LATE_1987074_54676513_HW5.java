import java.util.ArrayList;
import java.util.AbstractMap;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class HW5 {
    // joins events in date order for the display commands
    private static String format(List<Map.Entry<String, String>> events) {
        if (events.isEmpty()) return "none";
        StringBuilder result = new StringBuilder();
        for (Map.Entry<String, String> event : events) {
            if (result.length() > 0) result.append(' ');
            result.append(event.getKey()).append(':').append(event.getValue());
        }
        return result.toString();
    }

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length == 0) {
            System.err.println("usage: java HW5 <input file>");
            return;
        }
        SkipListMap timeline = new SkipListMap();
        Scanner input = new Scanner(new File(args[0]));

        // read each command and print its result
        while (input.hasNextLine()) {
            String line = input.nextLine().trim();
            if (line.isEmpty()) continue;
            String[] parts = line.split("\\s+", 3);
            String action = parts[0];
            switch (action) {
                case "AddEvent":
                    timeline.put(parts[1], parts[2]);
                    System.out.println(line + " success");
                    break;
                case "DisplayEvent":
                    String value = timeline.get(parts[1]);
                    System.out.println(line + " " + (value == null ? "none" : value));
                    break;
                case "DeleteEvent":
                    String removed = timeline.remove(parts[1]);
                    System.out.println(line + " " + (removed == null ? "noDateError" : "success"));
                    break;
                case "DisplayEventsBetweenDates":
                    System.out.println(line + " " + format(timeline.subMap(parts[1], parts[2])));
                    break;
                case "DisplayEventsFromStartDate":
                    System.out.println(line + " " + format(timeline.from(parts[1])));
                    break;
                case "DisplayEventsToEndDate":
                    System.out.println(line + " " + format(timeline.to(parts[1])));
                    break;
                case "DisplayAllEvents":
                    System.out.println(line + " " + format(timeline.all()));
                    break;
                case "PrintSkipList":
                    System.out.println(line);
                    System.out.println(timeline.printLevels());
                    break;
                default:
                    System.err.println("unknown action: " + action);
            }
        }
        input.close();
    }
}

class SkipListMap {
    private static final int MAX_HEIGHT = 10;

    private static class Node {
        String key;
        String value;
        Node[] next;

        Node(String key, String value, int height) {
            this.key = key;
            this.value = value;
            next = new Node[height + 1];
        }
    }

    private final Node head = new Node(null, null, MAX_HEIGHT);
    private final FakeRandHeight random = new FakeRandHeight();
    private int height = 0;

    // finds the nodes right before a key at each level
    private Node[] findPrevious(String key) {
        Node[] previous = new Node[MAX_HEIGHT + 1];
        Node current = head;
        for (int level = height; level >= 0; level--) {
            while (current.next[level] != null && current.next[level].key.compareTo(key) < 0) {
                current = current.next[level];
            }
            previous[level] = current;
        }
        return previous;
    }

    public String get(String key) {
        Node node = findPrevious(key)[0].next[0];
        return node != null && node.key.equals(key) ? node.value : null;
    }

    public String put(String key, String value) {
        Node[] previous = findPrevious(key);
        Node found = previous[0].next[0];
        if (found != null && found.key.equals(key)) {
            String old = found.value;
            found.value = value;
            return old;
        }

        // only new entries use the next random height
        int newHeight = Math.min(random.get(), MAX_HEIGHT - 1);
        if (newHeight > height) {
            for (int i = height + 1; i <= newHeight; i++) previous[i] = head;
            height = newHeight;
        }
        Node added = new Node(key, value, newHeight);
        for (int i = 0; i <= newHeight; i++) {
            added.next[i] = previous[i].next[i];
            previous[i].next[i] = added;
        }
        return null;
    }

    public String remove(String key) {
        Node[] previous = findPrevious(key);
        Node found = previous[0].next[0];
        if (found == null || !found.key.equals(key)) return null;
        for (int i = 0; i <= height; i++) {
            if (previous[i].next[i] == found) previous[i].next[i] = found.next[i];
        }
        // keep one empty level above the highest event level
        while (height > 0 && head.next[height] == null) height--;
        return found.value;
    }

    private Map.Entry<String, String> entry(Node node) {
        if (node == null || node == head) return null;
        return new AbstractMap.SimpleEntry<>(node.key, node.value);
    }

    public Map.Entry<String, String> floorEntry(String key) {
        Node before = findPrevious(key)[0];
        Node next = before.next[0];
        return entry(next != null && next.key.equals(key) ? next : before);
    }

    public Map.Entry<String, String> ceilingEntry(String key) {
        return entry(findPrevious(key)[0].next[0]);
    }

    public List<Map.Entry<String, String>> subMap(String start, String end) {
        List<Map.Entry<String, String>> results = new ArrayList<>();
        if (start.compareTo(end) > 0) return results;
        Map.Entry<String, String> first = ceilingEntry(start);
        Node current = first == null ? null : findPrevious(first.getKey())[0].next[0];
        while (current != null && current.key.compareTo(end) <= 0) {
            results.add(entry(current));
            current = current.next[0];
        }
        return results;
    }

    public List<Map.Entry<String, String>> from(String start) {
        List<Map.Entry<String, String>> results = new ArrayList<>();
        Node current = findPrevious(start)[0].next[0];
        while (current != null) {
            results.add(entry(current));
            current = current.next[0];
        }
        return results;
    }

    public List<Map.Entry<String, String>> to(String end) {
        return subMap("", end);
    }

    public List<Map.Entry<String, String>> all() {
        return from("");
    }

    public String printLevels() {
        StringBuilder result = new StringBuilder();
        // show an empty top level, like the sample output
        for (int level = height + 1; level >= 0; level--) {
            if (result.length() > 0) result.append('\n');
            result.append("(S").append(level).append(")");
            Node current = level <= height ? head.next[level] : null;
            if (current == null) result.append(" empty");
            while (current != null) {
                result.append(' ').append(current.key).append(':').append(current.value);
                current = current.next[level];
            }
        }
        return result.toString();
    }
}
