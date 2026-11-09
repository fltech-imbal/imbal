
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

public class HW5 {

    private static SkipList events = new SkipList();

    public static void main(String[] args) {

        if (args.length != 1) {
            System.err.println("Usage: java HW5 <filename>");
            return;
        }

        try (BufferedReader reader =
                     new BufferedReader(new FileReader(args[0]))) {

            String line;

            while ((line = reader.readLine()) != null) {
                line = line.trim();

                if (line.isEmpty()) {
                    continue;
                }

                processAction(line);
            }

        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        }
    }

    private static void processAction(String line) {

        String[] parts = line.split("\\s+", 3);
        String action = parts[0];

        switch (action) {

            case "DisplayEvent":
                if (parts.length >= 2) {
                    displayEvent(parts[1]);
                }
                break;

            case "AddEvent":
                if (parts.length >= 3) {
                    addEvent(parts[1], parts[2]);
                }
                break;

            case "DeleteEvent":
                if (parts.length >= 2) {
                    deleteEvent(parts[1]);
                }
                break;

            case "DisplayEventsBetweenDates":
                if (parts.length >= 3) {
                    displayEventsBetweenDates(
                        parts[1], parts[2]
                    );
                }
                break;

            case "DisplayEventsFromStartDate":
                if (parts.length >= 2) {
                    displayEventsFromStartDate(parts[1]);
                }
                break;

            case "DisplayEventsToEndDate":
                if (parts.length >= 2) {
                    displayEventsToEndDate(parts[1]);
                }
                break;

            case "DisplayAllEvents":
                displayAllEvents();
                break;

            case "PrintSkipList":
                System.out.println(events.printSkipList());
                break;

            default:
                System.err.println("Unknown action: " + action);
                break;
        }
    }

    private static int parseDate(String date) {
        return Integer.parseInt(date);
    }

    private static void displayEvent(String date) {

        int key = parseDate(date);
        String event = events.get(key);

        if (event == null) {
            System.out.println("none");
        } else {
            System.out.println(event);
        }
    }

    private static void addEvent(String date, String event) {

        int key = parseDate(date);

        events.put(key, event);

        System.out.println("success");
    }

    private static void deleteEvent(String date) {

        int key = parseDate(date);
        String removed = events.remove(key);

        if (removed == null) {
            System.out.println("noDateError");
        } else {
            System.out.println("success");
        }
    }

    private static void displayEventsBetweenDates(
            String startDate, String endDate) {

        int start = parseDate(startDate);
        int end = parseDate(endDate);

        List<SkipList.Entry> result = events.subMap(start, end);

        printEntries(result);
    }

    private static void displayEventsFromStartDate(
            String startDate) {

        int start = parseDate(startDate);

        SkipList.Entry first = events.ceilingEntry(start);

        if (first == null) {
            System.out.println("none");
            return;
        }

        List<SkipList.Entry> all = events.allEntries();

        StringBuilder result = new StringBuilder();

        for (SkipList.Entry entry : all) {

            if (entry.getKey() >= start) {

                if (result.length() > 0) {
                    result.append(" ");
                }

                result.append(entry);
            }
        }

        if (result.length() == 0) {
            System.out.println("none");
        } else {
            System.out.println(result);
        }
    }

    private static void displayEventsToEndDate(String endDate) {

        int end = parseDate(endDate);

        SkipList.Entry last = events.floorEntry(end);

        if (last == null) {
            System.out.println("none");
            return;
        }

        List<SkipList.Entry> all = events.allEntries();

        StringBuilder result = new StringBuilder();

        for (SkipList.Entry entry : all) {

            if (entry.getKey() > end) {
                break;
            }

            if (result.length() > 0) {
                result.append(" ");
            }

            result.append(entry);
        }

        if (result.length() == 0) {
            System.out.println("none");
        } else {
            System.out.println(result);
        }
    }

    private static void displayAllEvents() {

        List<SkipList.Entry> all = events.allEntries();

        printEntries(all);
    }

    private static void printEntries(
            List<SkipList.Entry> entries) {

        if (entries.isEmpty()) {
            System.out.println("none");
            return;
        }

        StringBuilder result = new StringBuilder();

        for (SkipList.Entry entry : entries) {

            if (result.length() > 0) {
                result.append(" ");
            }

            result.append(entry);
        }

        System.out.println(result);
    }
}
