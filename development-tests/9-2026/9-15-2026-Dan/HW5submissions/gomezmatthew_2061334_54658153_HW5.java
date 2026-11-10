/*

  Author: Matthew Gomez
  Course: CSE 2010
  Date: October 29, 2026
  Description of this file: Manages the timeline of major events of a
  user. Events are stored in a skip list (SkipListMap) whose key is the
  date (MMDD) and whose value is the event. The program reads a file of
  actions (AddEvent, DisplayEvent, DeleteEvent, DisplayEventsBetweenDates,
  DisplayEventsFromStartDate, DisplayEventsToEndDate, DisplayAllEvents,
  PrintSkipList) and prints one result per action to the standard output.
  Usage: java HW5 actionsFile

 */

import java.io.*;

public class HW5
{
    /* smallest and largest possible keys, used for the open-ended range queries */
    static final String EARLIEST = "0000";   // smaller than any MMDD date
    static final String LATEST = "9999";     // larger than any MMDD date

    /*
      Removes a byte order mark, zero-width spaces and extra spaces, so
      invisible characters in the input files cannot break the parsing.
      s: the line read from the file
      Returns the cleaned line.
    */
    static String clean(String s)
    {
        return s.replace("﻿", "").replace("​", "").replace(' ', ' ').trim();
    }

    /*
      Builds the text of the entries of a subMap result: "date1:event1 date2:event2 ...",
      or "none" if the list is empty. The list is emptied while it is read.
      entries: the entries returned by subMap
    */
    static String entriesToString(DoublyLinkedList<SkipListMap.Entry<String, String>> entries)
    {
        if (entries.isEmpty())
            return "none";
        StringBuilder sb = new StringBuilder();
        while (!entries.isEmpty())
        {
            SkipListMap.Entry<String, String> e = entries.removeFirst();
            if (sb.length() > 0) sb.append(" ");
            sb.append(e.getKey()).append(":").append(e.getValue());
        }
        return sb.toString();
    }

    /*
      Prints every list of the skip list, from the top list down to S0.
      timeline: the skip list that holds the events
    */
    static void printSkipList(SkipListMap<String, String> timeline)
    {
        for (int level = timeline.getHeight(); level >= 0; level--)
            System.out.println(timeline.levelToString(level));
    }

    /*
      args[0]: the file of actions, one action per line
    */
    public static void main(String[] args) throws IOException
    {
        /* timeline: skip list of events (key = date MMDD, value = event);
           randHeight: gives the height of each new entry;
           in: reader of the actions file; line: the current (cleaned) line;
           tk: the words of the current line; action: the first word of the line */
        if (args.length < 1)
        {
            System.out.println("Usage: java HW5 actionsFile");
            return;
        }
        FakeRandHeight randHeight = new FakeRandHeight();
        SkipListMap<String, String> timeline = new SkipListMap<String, String>(randHeight);
        BufferedReader in = new BufferedReader(new FileReader(args[0]));
        String line;

        /* read the actions one line at a time, perform each one and print its result */
        while ((line = in.readLine()) != null)
        {
            line = clean(line);
            if (line.isEmpty()) continue;
            String[] tk = line.split("\\s+", 3);   // action, first argument, rest of the line
            String action = tk[0];

            if (action.equals("AddEvent") && tk.length >= 3)
            {
                /* add the event (a date that already exists gets its event replaced) */
                timeline.put(tk[1], tk[2]);
                System.out.println("AddEvent " + tk[1] + " " + tk[2] + " success");
            }
            else if (action.equals("DisplayEvent") && tk.length >= 2)
            {
                /* print the event of the date, or none */
                String event = timeline.get(tk[1]);
                System.out.println("DisplayEvent " + tk[1] + " " + (event == null ? "none" : event));
            }
            else if (action.equals("DeleteEvent") && tk.length >= 2)
            {
                /* remove the event of the date; noDateError if the date is not in the timeline */
                String old = timeline.remove(tk[1]);
                System.out.println("DeleteEvent " + tk[1] + " " + (old == null ? "noDateError" : "success"));
            }
            else if (action.equals("DisplayEventsBetweenDates") && tk.length >= 3)
            {
                /* all events with startDate <= date <= endDate */
                System.out.println("DisplayEventsBetweenDates " + tk[1] + " " + tk[2] + " "
                        + entriesToString(timeline.subMap(tk[1], tk[2])));
            }
            else if (action.equals("DisplayEventsFromStartDate") && tk.length >= 2)
            {
                /* all events with date >= startDate */
                System.out.println("DisplayEventsFromStartDate " + tk[1] + " "
                        + entriesToString(timeline.subMap(tk[1], LATEST)));
            }
            else if (action.equals("DisplayEventsToEndDate") && tk.length >= 2)
            {
                /* all events with date <= endDate */
                System.out.println("DisplayEventsToEndDate " + tk[1] + " "
                        + entriesToString(timeline.subMap(EARLIEST, tk[1])));
            }
            else if (action.equals("DisplayAllEvents"))
            {
                /* every event, in date order */
                System.out.println("DisplayAllEvents " + entriesToString(timeline.subMap(EARLIEST, LATEST)));
            }
            else if (action.equals("PrintSkipList"))
            {
                /* print the lists from the top one down to S0 */
                System.out.println("PrintSkipList");
                printSkipList(timeline);
            }
        }
        in.close();
    }
}
