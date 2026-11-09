/*
 Author: Parker Blue
 Email: pblue2025@my.fit.edu
 Course: CSE 2010
 Section: 9:30 AM Lab
 Description: Manages a user's timeline of events keyed by date using a skip list. Reads actions
 from the file named on the command line and prints one result per action to standard output.
*/
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class HW5 {
  /* the main function is the filename of actions where each line is parsed and dispatched to the
  matching skip list operation.
  */
  public static void main(String[] args) throws IOException {
    // events will show the date and then the event description
    SkipListMap<String, String> events = new SkipListMap<>();
    // reads each action line and then processes it
    try (BufferedReader in = new BufferedReader(new FileReader(args[0]))) {
      String line;
      while ((line = in.readLine()) != null) {
        line = line.trim();
        if (line.isEmpty()) continue;
        processAction(line, events);
      }
    }
  }

  /* processAction will parse one action line and then print its result
  line - the action line from the input file
  events - the skip list of events
  */
  private static void processAction(String line, SkipListMap<String, String> events) {
    String[] tok = line.split("\\s+", 3);
    String action = tok[0];
    switch (action) {
      case "AddEvent":
        events.put(tok[1], tok[2].trim());
        System.out.println("AddEvent " + tok[1] + " " + tok [2].trim() + " success");
        break;
      case "DisplayEvent": {
        String ev = events.get(tok[1]);
        System.out.println("DisplayEvent " + tok[1] + " " + (ev == null ? "none" : ev));
        break;
      }
      case "DeleteEvent": {
        String ev = events.remove(tok[1]);
        System.out.println("DeleteEvent " + tok[1] + " " + (ev == null ? "noDateError" : "success"));
        break;
      }
      case "DisplayEventsBetweenDates": {
        String[] t = line.split("\\s+");
        System.out.println("DisplayEventsBetweenDates " + t[1] + " " + t[2] + " " + format(events.subMap(t[1], t[2])));
        break;
      }
      case "DisplayEventsFromStartDate": {
        SkipListMap.Entry<String, String> last = events.lastEntry();
        ArrayList<SkipListMap.Entry<String, String>> list = (last == null) ? new ArrayList<>() : events.subMap(tok[1], last.getKey());
        System.out.println("DisplayEventsFromStartDate " + tok[1] + " " + format(list));
        break;
      }
      case "DisplayEventsToEndDate": {
        SkipListMap.Entry<String, String> first = events.firstEntry();
        ArrayList<SkipListMap.Entry<String, String>> list = (first == null) ? new ArrayList<>() : events.subMap(first.getKey(), tok[1]);
        System.out.println("DisplayEventsToEndDate " + tok[1] + " " + format(list));
        break;
      }
      case "DisplayAllEvents": {
        SkipListMap.Entry<String, String> first = events.firstEntry();
        SkipListMap.Entry<String, String> last = events.lastEntry();
        ArrayList<SkipListMap.Entry<String, String>> list = (first == null) ? new ArrayList<>() : events.subMap(first.getKey(), last.getKey());
        System.out.println("DisplayAllEvents " + format(list));
        break;
      }
      case "PrintSkipList":
        System.out.println("PrintSkipList");
        for (String s : events.levelStrings()) {
          System.out.println(s);
        }
        break;
      default:
        System.out.println(line + " invalidAction");
    }
  }

  /* format is to show the date and then event in order or say "none" if empty
  list is the entries to be formatted in key order
  */
  private static String format(ArrayList<SkipListMap.Entry<String, String>> list) {
    if (list.isEmpty()) return "none";
    StringBuilder sb = new StringBuilder();
    for (SkipListMap.Entry<String, String> e : list) {
      if (sb.length() > 0) sb.append(' ');
      sb.append(e.getKey()).append(':').append(e.getValue());
    }
    return sb.toString();
  }
}
