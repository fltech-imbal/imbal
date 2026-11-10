/*
  Author: ( Carson Rocha)
  Email: (Crocha2025@my.fit.edu)
  Course: CSE 2010
  Section: (930 am)
  Description: Stores timeline events in a skip list and processes
               commands from a text file.
*/

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class HW5 {

    // One date and its event, used when returning search results.
    private static class Entry {
        // Save the date and event text.
        String date;
        String event;

        Entry(String date, String event) {
            this.date = date;
            this.event = event;
        }

        // Show an event as date:event.
        public String toString() {
            return date + ":" + event;
        }
    }

    // A skip list that stores dates in ascending order.
    private static class SkipListMap {
        // The tallest allowed skip-list level is 10.
        private static final int MAX_HEIGHT = 10;

        // A node can link to its neighbors and to the levels above/below it.
        private static class Node {
            String date;
            String event;
            // These links let us move in four directions.
            Node next;
            Node prev;
            Node up;
            Node down;

            Node(String date, String event) {
                this.date = date;
                this.event = event;
            }
        }

        // Keep the start of every level in this array.
        private Node[] heads = new Node[MAX_HEIGHT + 2];
        // Use the given height generator for new dates.
        private FakeRandHeight randHeight = new FakeRandHeight();

        // Make an empty head node for each level.
        SkipListMap() {
            for (int level = 0; level < heads.length; level++) {
                heads[level] = new Node(null, null);
                // Join this empty level to the one below it.
                if (level > 0) {
                    heads[level].down = heads[level - 1];
                    heads[level - 1].up = heads[level];
                }
            }
        }

        // Find the last bottom-level node whose date is smaller than date.
        private Node findBefore(String date) {
            // Start at the top and search toward the bottom.
            Node current = heads[MAX_HEIGHT];
            for (int level = MAX_HEIGHT; level >= 0; level--) {
                // Move right while the next date is too early.
                while (current.next != null &&
                       current.next.date.compareTo(date) < 0) {
                    current = current.next;
                }
                // Move down to keep searching.
                if (level > 0) {
                    current = current.down;
                }
            }
            return current;
        }

        // Return the event for this date, or null if it is not present.
        String get(String date) {
            // Check whether the next date matches exactly.
            Node next = findBefore(date).next;
            if (next != null && next.date.equals(date)) {
                return next.event;
            }
            return null;
        }

        // Insert a new date, or replace an existing event.
        // Return null for a new date, or the old event for a replacement.
        String put(String date, String event) {
            // Save where the new date belongs on each level.
            Node[] before = new Node[MAX_HEIGHT + 1];
            Node current = heads[MAX_HEIGHT];

            // Remember the insertion position at every level.
            for (int level = MAX_HEIGHT; level >= 0; level--) {
                while (current.next != null &&
                       current.next.date.compareTo(date) < 0) {
                    current = current.next;
                }
                // Remember the node just before the new date.
                before[level] = current;
                if (level > 0) {
                    current = current.down;
                }
            }

            // If the date already exists, update its entire tower.
            Node existing = before[0].next;
            if (existing != null && existing.date.equals(date)) {
                // Save the old event before replacing it.
                String oldEvent = existing.event;
                while (existing != null) {
                    existing.event = event;
                    existing = existing.up;
                }
                return oldEvent;
            }

            // Use the given generator to pick how many levels to use.
            int height = Math.min(randHeight.get(), MAX_HEIGHT);
            Node below = null;
            for (int level = 0; level <= height; level++) {
                // Make one copy of the event on this level.
                Node newNode = new Node(date, event);
                Node left = before[level];
                Node right = left.next;

                // Insert the new node between its two neighbors.
                newNode.prev = left;
                newNode.next = right;
                left.next = newNode;
                if (right != null) {
                    right.prev = newNode;
                }

                // Connect this node to the same date on the level below.
                newNode.down = below;
                if (below != null) {
                    below.up = newNode;
                }
                below = newNode;
            }
            return null;
        }

        // Remove the date from every level, returning its event if found.
        String remove(String date) {
            Node current = findBefore(date).next;
            // There is nothing to delete if the date is missing.
            if (current == null || !current.date.equals(date)) {
                return null;
            }

            // Remove the event from every level it uses.
            String oldEvent = current.event;
            while (current != null) {
                current.prev.next = current.next;
                if (current.next != null) {
                    current.next.prev = current.prev;
                }
                // Go up to remove the next copy.
                current = current.up;
            }
            return oldEvent;
        }

        // Get the largest entry whose date is <= the requested date.
        Entry floorEntry(String date) {
            Node before = findBefore(date);
            Node next = before.next;
            // An exact match is also the floor entry.
            if (next != null && next.date.equals(date)) {
                return new Entry(next.date, next.event);
            }
            // Otherwise, use the closest earlier date.
            if (before.date != null) {
                return new Entry(before.date, before.event);
            }
            return null;
        }

        // Get the smallest entry whose date is >= the requested date.
        Entry ceilingEntry(String date) {
            Node next = findBefore(date).next;
            // No later date means there is no ceiling entry.
            if (next == null) {
                return null;
            }
            return new Entry(next.date, next.event);
        }

        // Return entries between start and end, including both dates.
        // A null bound means there is no limit on that side.
        List<Entry> subMap(String start, String end) {
            List<Entry> result = new ArrayList<Entry>();
            // A backwards date range has no results.
            if (start != null && end != null && start.compareTo(end) > 0) {
                return result;
            }

            Node current;
            // Start from the first date when no start is given.
            if (start == null) {
                current = heads[0].next;
            } else {
                current = findBefore(start).next;
            }

            // Keep adding dates until the end of the range.
            while (current != null &&
                   (end == null || current.date.compareTo(end) <= 0)) {
                result.add(new Entry(current.date, current.event));
                current = current.next;
            }
            return result;
        }

        // Print the skip list from the empty top level down to level 0.
        void printSkipList() {
            // Find the highest level that has an event.
            int highest = -1;
            for (int level = MAX_HEIGHT; level >= 0; level--) {
                if (heads[level].next != null) {
                    highest = level;
                    break;
                }
            }

            int top = highest + 1; // One empty level above the tallest tower.
            for (int level = top; level >= 0; level--) {
                // Start each output line with its level number.
                StringBuilder line = new StringBuilder("(S" + level + ")");
                Node current = heads[level].next;
                // Show empty when this level has no dates.
                if (current == null) {
                    line.append(" empty");
                } else {
                    while (current != null) {
                        line.append(" ").append(current.date)
                            .append(":").append(current.event);
                        current = current.next;
                    }
                }
                System.out.println(line);
            }
        }
    }

    // Put the returned events on one line, separated by spaces.
    private static String formatEvents(List<Entry> entries) {
        // Display none when no events were found.
        if (entries.isEmpty()) {
            return "none";
        }
        StringBuilder result = new StringBuilder();
        for (Entry entry : entries) {
            // Put spaces between displayed events.
            if (result.length() > 0) {
                result.append(" ");
            }
            result.append(entry);
        }
        return result.toString();
    }

    // Read each command in the input file and print the expected response.
    public static void main(String[] args) {
        // Make sure an input filename was provided.
        if (args.length == 0) {
            System.err.println("Usage: java HW5 <inputFile>");
            return;
        }

        SkipListMap events = new SkipListMap();
        try (BufferedReader reader = new BufferedReader(new FileReader(args[0]))) {
            String line;
            // Read and handle one command at a time.
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                // Skip blank lines in the file.
                if (line.isEmpty()) {
                    continue;
                }

                // Split into command, date, and remaining event text.
                String[] parts = line.split("\\s+", 3);
                String command = parts[0];

                switch (command) {
                    // Add a date and its event.
                    case "AddEvent": {
                        String date = parts[1];
                        String event = parts[2];
                        events.put(date, event);
                        System.out.println("AddEvent " + date + " " + event + " success");
                        break;
                    }
                    // Remove an event using its date.
                    case "DeleteEvent": {
                        String date = parts[1];
                        String deleted = events.remove(date);
                        System.out.println("DeleteEvent " + date + " " +
                                (deleted == null ? "noDateError" : "success"));
                        break;
                    }
                    // Look up one event.
                    case "DisplayEvent": {
                        String date = parts[1];
                        String event = events.get(date);
                        System.out.println("DisplayEvent " + date + " " +
                                (event == null ? "none" : event));
                        break;
                    }
                    // Show events within both dates.
                    case "DisplayEventsBetweenDates": {
                        String start = parts[1];
                        String end = parts[2];
                        System.out.println(command + " " + start + " " + end + " " +
                                formatEvents(events.subMap(start, end)));
                        break;
                    }
                    // Show events on or after this date.
                    case "DisplayEventsFromStartDate": {
                        String start = parts[1];
                        System.out.println(command + " " + start + " " +
                                formatEvents(events.subMap(start, null)));
                        break;
                    }
                    // Show events on or before this date.
                    case "DisplayEventsToEndDate": {
                        String end = parts[1];
                        System.out.println(command + " " + end + " " +
                                formatEvents(events.subMap(null, end)));
                        break;
                    }
                    // Show every event in order.
                    case "DisplayAllEvents": {
                        System.out.println(command + " " +
                                formatEvents(events.subMap(null, null)));
                        break;
                    }
                    // Display each level of the skip list.
                    case "PrintSkipList": {
                        System.out.println(command);
                        events.printSkipList();
                        break;
                    }
                    default:
                        System.err.println("Unknown command: " + command);
                }
            }
        } catch (IOException exception) {
            // Report a problem if the file cannot be read.
            System.err.println("Unable to read input file: " + exception.getMessage());
        }
    }
}