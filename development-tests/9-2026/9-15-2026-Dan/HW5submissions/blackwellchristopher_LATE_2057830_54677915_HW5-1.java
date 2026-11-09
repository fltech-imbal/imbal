/*

  Author: Christopher Blackwell
  Email: cblackwell2025@my.fit.edu
  Course: CSE2010
  Section: 3:30-4:45pm
  Description of this file: sort photos from given file list

 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class HW5 {
    private static final int MAX_LEVEL = 10;
    private final SkipListMap events = new SkipListMap();

    private static final class EventEntry {
        final String date;
        String event;

        EventEntry(String date, String event) {
            this.date = date;
            this.event = event;
        }
    }

    private static final class SkipNode {
        EventEntry entry;
        SkipNode[] next;

        SkipNode(int height, EventEntry entry) {
            this.entry = entry;
            this.next = new SkipNode[height + 1];
        }
    }

   
    private static final class SkipListMap {
        private final SkipNode head = new SkipNode(MAX_LEVEL, null);
        private final FakeRandHeight randHeight = new FakeRandHeight();
        private int size;

        EventEntry get(String key) {
            SkipNode node = predecessor(key, 0).next[0];
            return node != null && node.entry.date.equals(key) ? node.entry : null;
        }

        EventEntry put(String key, String value) {
            SkipNode[] update = new SkipNode[MAX_LEVEL + 1];
            SkipNode current = head;
            for (int level = MAX_LEVEL; level >= 0; level--) {
                while (current.next[level] != null
                        && current.next[level].entry.date.compareTo(key) < 0) {
                    current = current.next[level];
                }
                update[level] = current;
            }

            current = current.next[0];
            if (current != null && current.entry.date.equals(key)) {
                String old = current.entry.event;
                current.entry.event = value;
                return new EventEntry(key, old);
            }

            int height = Math.min(randHeight.get(), MAX_LEVEL);
            SkipNode inserted = new SkipNode(height, new EventEntry(key, value));
            for (int level = 0; level <= height; level++) {
                inserted.next[level] = update[level].next[level];
                update[level].next[level] = inserted;
            }
            size++;
            return null;
        }

        EventEntry remove(String key) {
            SkipNode[] update = new SkipNode[MAX_LEVEL + 1];
            SkipNode current = head;
            for (int level = MAX_LEVEL; level >= 0; level--) {
                while (current.next[level] != null
                        && current.next[level].entry.date.compareTo(key) < 0) {
                    current = current.next[level];
                }
                update[level] = current;
            }

            current = current.next[0];
            if (current == null || !current.entry.date.equals(key)) return null;
            for (int level = 0; level < current.next.length; level++) {
                if (update[level].next[level] == current) {
                    update[level].next[level] = current.next[level];
                }
            }
            size--;
            return current.entry;
        }

        private SkipNode predecessor(String key, int level) {
            SkipNode current = head;
            while (current.next[level] != null
                    && current.next[level].entry.date.compareTo(key) < 0) {
                current = current.next[level];
            }
            return current;
        }

        List<EventEntry> between(String start, String end) {
            List<EventEntry> result = new ArrayList<>();
            if (start.compareTo(end) > 0) return result;
            SkipNode current = predecessor(start, 0).next[0];
            while (current != null && current.entry.date.compareTo(end) <= 0) {
                result.add(current.entry);
                current = current.next[0];
            }
            return result;
        }

        List<EventEntry> from(String start) {
            List<EventEntry> result = new ArrayList<>();
            SkipNode current = predecessor(start, 0).next[0];
            while (current != null) {
                result.add(current.entry);
                current = current.next[0];
            }
            return result;
        }

        List<EventEntry> all() {
            return from("");
        }

        String printSkipList() {
            StringBuilder output = new StringBuilder();
            for (int level = 3; level >= 0; level--) {
                output.append("(S").append(level).append(") ");
                SkipNode current = head.next[level];
                if (current == null) {
                    output.append("empty");
                } else {
                    boolean first = true;
                    while (current != null) {
                        if (!first) output.append(' ');
                        output.append(current.entry.date).append(':').append(current.entry.event);
                        first = false;
                        current = current.next[level];
                    }
                }
                output.append(System.lineSeparator());
            }
            return output.toString();
        }
    }

    private static String entriesText(List<EventEntry> entries) {
        if (entries.isEmpty()) return "none";
        StringBuilder result = new StringBuilder();
        for (EventEntry entry : entries) {
            if (result.length() > 0) result.append(' ');
            result.append(entry.date).append(':').append(entry.event);
        }
        return result.toString();
    }

    private void process(String line) {
        line = line.trim();
        if (line.isEmpty()) return;

        String[] parts = line.split("\\s+", 3);
        String action = parts[0];
        try {
            switch (action) {
                case "AddEvent": {
                    if (parts.length < 3) return;
                    String date = parts[1];
                    String event = parts[2];
                    events.put(date, event);
                    System.out.println("AddEvent " + date + " " + event + " success");
                    break;
                }
                case "DisplayEvent": {
                    if (parts.length < 2) return;
                    String date = parts[1];
                    EventEntry entry = events.get(date);
                    System.out.println("DisplayEvent " + date + " "
                            + (entry == null ? "none" : entry.event));
                    break;
                }
                case "DeleteEvent": {
                    if (parts.length < 2) return;
                    String date = parts[1];
                    EventEntry removed = events.remove(date);
                    System.out.println("DeleteEvent " + date + " "
                            + (removed == null ? "noDateError" : "success"));
                    break;
                }
                case "DisplayEventsBetweenDates": {
                    if (parts.length < 3) return;
                    String[] args = parts[2].split("\\s+");
                    if (args.length < 1) return;
                    System.out.println("DisplayEventsBetweenDates " + parts[1] + " " + args[0]
                            + " " + entriesText(events.between(parts[1], args[0])));
                    break;
                }
                case "DisplayEventsFromStartDate": {
                    if (parts.length < 2) return;
                    System.out.println("DisplayEventsFromStartDate " + parts[1] + " "
                            + entriesText(events.from(parts[1])));
                    break;
                }
                case "DisplayEventsToEndDate": {
                    if (parts.length < 2) return;
                    System.out.println("DisplayEventsToEndDate " + parts[1] + " "
                            + entriesText(events.between("0000", parts[1])));
                    break;
                }
                case "DisplayAllEvents":
                    System.out.println("DisplayAllEvents " + entriesText(events.all()));
                    break;
                case "PrintSkipList":
                    System.out.print("PrintSkipList" + System.lineSeparator());
                    System.out.print(events.printSkipList());
                    break;
                default:
                    break;
            }
        } catch (RuntimeException ex) {
            System.err.println("Unable to process action: " + line);
        }
    }

    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Usage: java HW5 filename");
            return;
        }
        HW5 program = new HW5();
        try (BufferedReader reader = new BufferedReader(new FileReader(args[0]))) {
            String line;
            while ((line = reader.readLine()) != null) program.process(line);
        } catch (IOException e) {
            System.err.println("Error reading input file: " + e.getMessage());
        }
    }
}
