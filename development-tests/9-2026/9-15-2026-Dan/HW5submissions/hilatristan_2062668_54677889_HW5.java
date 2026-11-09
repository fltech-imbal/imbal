/*
 * Author: Tristan Hila
 * Email: thila2025@my.fit.edu
 * Course: CSE 2010
 * Section: Lab 3:30 to 4:45
 * Description: Manage a timeline of dated events using a skip list.
 */

import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class HW5 {
    /** Format an integer date with the four digits required by MMDD. */
    private static String date(int key) {
        return String.format(java.util.Locale.ROOT, "%04d", key);
    }

    /** Append the sorted entries, or none when the inclusive range is empty. */
    private static String events(List<SkipListMap.Entry<Integer, String>> entries) {
        if (entries.isEmpty()) return " none";
        StringBuilder result = new StringBuilder();
        for (SkipListMap.Entry<Integer, String> entry : entries) {
            result.append(' ').append(date(entry.getKey()))
                  .append(':').append(entry.getValue());
        }
        return result.toString();
    }

    /** Read actions from args[0] and print the result of each action. */
    public static void main(String[] args) throws java.io.IOException {
        if (args.length != 1) {
            System.err.println("Usage: java HW5 filename");
            return;
        }
        SkipListMap<Integer, String> timeline = new SkipListMap<>();
        try (BufferedReader input = Files.newBufferedReader(
                Paths.get(args[0]), StandardCharsets.UTF_8)) {
            String line;
            while ((line = input.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                // A limit of three preserves spaces within an event's name.
                String[] fields = line.split("\\s+", 3);
                String action = fields[0];
                int key = fields.length > 1 ? Integer.parseInt(fields[1]) : 0;
                String prefix = action + (fields.length > 1 ? " " + date(key) : "");

                switch (action) {
                    case "AddEvent":
                        timeline.put(key, fields[2]);
                        System.out.println(prefix + " " + fields[2] + " success");
                        break;
                    case "DisplayEvent":
                        String event = timeline.get(key);
                        System.out.println(prefix + " " + (event == null ? "none" : event));
                        break;
                    case "DeleteEvent":
                        System.out.println(prefix + (timeline.remove(key) == null
                                ? " noDateError" : " success"));
                        break;
                    case "DisplayEventsBetweenDates":
                        int end = Integer.parseInt(fields[2].trim());
                        System.out.println(prefix + " " + date(end)
                                + events(timeline.subMap(key, end)));
                        break;
                    case "DisplayEventsFromStartDate":
                        System.out.println(prefix + events(timeline.subMap(key, Integer.MAX_VALUE)));
                        break;
                    case "DisplayEventsToEndDate":
                        System.out.println(prefix + events(timeline.subMap(Integer.MIN_VALUE, key)));
                        break;
                    case "DisplayAllEvents":
                        System.out.println(action + events(timeline.subMap(
                                Integer.MIN_VALUE, Integer.MAX_VALUE)));
                        break;
                    case "PrintSkipList":
                        System.out.println(action);
                        System.out.println(timeline.formatLevels(HW5::date));
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown action: " + action);
                }
            }
        }
    }
}
