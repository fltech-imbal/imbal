/*
Author: Airen Sharber
Email: asharber2025@code01.fit.edu
Course: CSE2010
Section: 5
Description of this file:
*/
import java.util.*;
import java.io.*;
public class HW5 {
  static class Node {
    String date;
    String event;
    Node[] next;
    Node(String date, String event, int height) {
      this.date = date;
      this.event = event;
      next = new Node[height + 1];
    }
  }
  static class SkipListMap {
    private final Node head = new Node(null, null, 10);
    private final FakeRandHeight randomHeight = new FakeRandHeight();
    private int highest = 0;
    private Node[] predecessors(String date) {
      Node[] prev = new Node[11];
      Node current = head;
      for (int level = highest; level >= 0; level--) {
        while (current.next[level] != null && 
               current.next[level].date.compareTo(date) < 0) {
          current = current.next[level];
        }
        prev[level] = current;
      }
      return prev;
    }
    String put(String date, String event) {
      Node[] prev = predecessors(date);
      Node found = prev[0].next[0];
      if (found != null && found.date.equals(date)) {
        String old = found.event;
        found.event = event;
        return old;
      }
      int height = randomHeight.get();
      if (height > 10) {
        height = 10;
      }
      if (height > highest) {
        for (int i = highest+1; i <= height; i++) {
          prev[i] = head;
        }
        highest = height;
      }
      Node added = new Node(date, event, height);
      for (int i = 0; i <= height; i++) {
        added.next[i] = prev[i].next[i];
        prev[i].next[i] = added;
      }
      return null;
    }
    String remove(String date) {
      Node[] prev = predecessors(date);
      Node found = prev[0].next[0];
      if (found == null || !found.date.equals(date)) {
        return null;
      }
      for (int i = 0; i < found.next.length; i++) {
        prev[i].next[i] = found.next[i];
      }
      while (highest > 0 && head.next[highest] == null) {
        highest--;
      }
      return found.event;
    }
    Node floorEntry(String date) {
      Node[] prev = predecessors(date);
      Node next = prev[0].next[0];
      if (next != null && next.date.equals(date)) {
        return next;
      }
      if (prev[0] == head) {
        return null;
      } else {
        return prev[0];
      }
    }
    Node ceilingEntry(String date) {
      return predecessors(date)[0].next[0];
    }
    ArrayList<Node> subMap (String first, String last) {
      ArrayList<Node> result = new ArrayList<Node>();
      Node current = ceilingEntry(first);
      while (current != null && current.date.compareTo(last) <= 0) {
        result.add(current);
        current = current.next[0];
      }
      return result;
    }
    ArrayList<Node> fromDate(String date) {
      ArrayList<Node> result = new ArrayList<Node>();
      Node current = ceilingEntry(date);
      while (current != null) {
        result.add(current);
        current = current.next[0];
      }
      return result;
    }
    ArrayList<Node> toDate(String date) {
      ArrayList<Node> result = new ArrayList<Node>();
      Node current = head.next[0];
      while (current != null && current.date.compareTo(date) <= 0) {
        result.add(current);
        current = current.next[0];
      }
      return result;
    }
    ArrayList<Node> all() {
      ArrayList<Node> result = new ArrayList<Node>();
      Node current = head.next[0];
      while (current != null) {
        result.add(current);
        current = current.next[0];
      }
      return result;
    }
    void printSkipList() {
      System.out.println("PrintSkipList");
      for (int i = highest+1; i >= 0; i--) {
        String line = "(S" + i + ")";
        Node current;
        if (i <= 10) {
          current = head.next[i];
        } else {
          current = null;
        }
        if (current == null) {
          line = line + " empty";
        } else {
          while (current != null) {
            line = line + " " + current.date + ":" + current.event;
            current = current.next[i];
          }
        }
        System.out.println(line);
      }
    }
    String get(String date) {
      Node[] prev = predecessors(date);
      Node found = prev[0].next[0];
      if (found != null && found.date.equals(date)) {
        return found.event;
      }
      return null;
    }
  }
  private static void printEvents(String command, ArrayList<Node> events) {
    String line = command;
    if (events.isEmpty()) {
      line = line + " none";
    } else {
      for (Node event : events) {
        line = line + " " + event.date + ":" + event.event;
      }
    }
    System.out.println(line);
  }
  public static void main(String[] args) throws FileNotFoundException {
    if (args.length == 0) {
      System.out.println("Usage: java HW5 filename");
      return;
    }
    SkipListMap timeline = new SkipListMap();
    Scanner input = new Scanner(new File(args[0]));
    while (input.hasNextLine()) {
      String line = input.nextLine().trim();
      if (line.isEmpty()) {
        continue;
      }
      String[] parts = line.split("\\s+", 3);
      String command = parts[0];
      switch (command) {
        case "AddEvent":
          timeline.put(parts[1], parts[2]);
          System.out.println("AddEvent " + parts[1] + " " + parts[2] + " success");
          break;
        case "DisplayEvent":
          String event = timeline.get(parts[1]);
          System.out.println("DisplayEvent " + parts[1] + " " + (event == null ? "none" : event));
          break;
        case "DeleteEvent":
          String removed = timeline.remove(parts[1]);
          System.out.println("DeleteEvent " + parts[1]
                      + " " + (removed == null ? "noDateError" : "success"));
          break;
        case "DisplayEventsBetweenDates":
          printEvents(line,timeline.subMap(parts[1], parts[2]));
          break;
        case "DisplayEventsFromStartDate":
          printEvents(line, timeline.fromDate(parts[1]));
          break;
        case "DisplayEventsToEndDate":
          printEvents(line, timeline.toDate(parts[1]));
          break;
        case "DisplayAllEvents":
          printEvents(command, timeline.all());
          break;
        case "PrintSkipList":
          timeline.printSkipList();
          break;
        default:
          System.err.println("Unknown command: " + command);
       }
    }
  }    
}
