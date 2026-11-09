/*
  Author: Jack Young
  Email: Jack2025@my.fit.edu
  Course: CSE2010
  Section: 1
  Description: Manages a timeline of the major events of its users from a time range using a skip list.
*/
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;

public class HW5
{
    static class Node
    {
        String date;
        String event;
        Node[] next;

        Node(String date, String event, int height)
        {
            this.date = date;
            this.event = event;
            next = new Node[height + 1];
        }
    }

    static class SkipListMap
    {
        Node head;
        int height;
        FakeRandHeight randHeight;

        SkipListMap()
        {
            head = new Node("", "", 9);
            height = 0;
            randHeight = new FakeRandHeight();
        }

        public String get(String date)
        {
            Node current = head;

            for (int i = height; i >= 0; i--)
            {
                while (current.next[i] != null && current.next[i].date.compareTo(date) < 0)
                {
                    current = current.next[i];
                }
            }

            current = current.next[0];

            if (current != null && current.date.equals(date)) {
                return current.event;
            }

            return null;
        }

        public String put(String date, String event)
        {
            Node[] previous = new Node[10];
            Node current = head;

            for (int i = height; i >= 0; i--)
            {
                while (current.next[i] != null &&
                       current.next[i].date.compareTo(date) < 0)
                {
                    current = current.next[i];
                }

                previous[i] = current;
            }

            current = current.next[0];

            if (current != null && current.date.equals(date))
            {
                String oldevent = current.event;
                current.event = event;
                return oldevent;
            }

            int newHeight = randHeight.get();

            if (newHeight > height)
            {
                for (int i = height + 1; i <= newHeight; i++) {
                    previous[i] = head;
                }

                height = newHeight;
            }

            Node newest = new Node(date, event, newHeight);

            for (int i = 0; i <= newHeight; i++) 
            {
                newest.next[i] = previous[i].next[i];
                previous[i].next[i] = newest;
            }

            return null;
        }

        public String remove(String date)
        {
            Node[] previous = new Node[10];
            Node current = head;

            for (int i = height; i >= 0; i--)
            {
                while (current.next[i] != null && current.next[i].date.compareTo(date) < 0) 
                {
                    current = current.next[i];
                }
                previous[i] = current;
            }

            current = current.next[0];

            if (current == null || !current.date.equals(date)) 
            {
                return null;
            }

            for (int i = 0; i <= height; i++)
            {
                if (previous[i].next[i] == current) {
                    previous[i].next[i] = current.next[i];
                }
            }

            while (height > 0 && head.next[height] == null) 
            {
                height--;
            }

            return current.event;
        }

        public Node floorEntry(String date)
        {
            Node current = head;

            for (int i = height; i >= 0; i--)
            {
                while (current.next[i] != null && current.next[i].date.compareTo(date) <= 0)
                {
                    current = current.next[i];
                }
            }

            if (current == head) {
                return null;
            }
            return current;
        }

        public Node ceilingEntry(String date)
        {
            Node current = head;

            for (int i = height; i >= 0; i--)
            {
                while (current.next[i] != null && current.next[i].date.compareTo(date) < 0)
                {
                    current = current.next[i];
                }
            }

            return current.next[0];
        }

        public String subMap(String date1, String date2)
        {
            String result = "";
            Node current = ceilingEntry(date1);

            while (current != null && current.date.compareTo(date2) <= 0)
            {
                if (!result.equals("")) {
                    result += " ";
                }
                result += current.date + ":" + current.event;
                current = current.next[0];
            }

            if (result.equals("")) {
                return "none";
            }
            return result;
        }

        public String allEvents()
        {
            return subMap("0000", "9999");
        }

       public void printSkipList()
        {
            for (int i = height + 1; i >= 0; i--)
            {
                System.out.print("(S" + i + ")");

                Node current = head.next[i];

                if (current == null) {
                    System.out.println(" empty");
                } else {
                    while (current != null)
                    {
                        System.out.print(" " + current.date + ":" + current.event);
                        current = current.next[i];
                    }

                    System.out.println();
                }
            }
        }
    }

    public static void main(String[] args) throws IOException
     {
        BufferedReader reader = new BufferedReader(new FileReader(args[0]));

        SkipListMap events = new SkipListMap();
        String line;

        while ((line = reader.readLine()) != null)
        {
            StringTokenizer input = new StringTokenizer(line);
            String action = input.nextToken();

            if (action.equals("AddEvent"))
            {
                String date = input.nextToken();
                String event = input.nextToken();
                events.put(date, event);
                System.out.println("AddEvent " + date + " " + event + " success");
            }
            
            else if (action.equals("DisplayEvent"))
            {
                String date = input.nextToken();
                String event = events.get(date);

                if (event == null) {
                    event = "none";
                }

                System.out.println("DisplayEvent " + date + " " + event);
            }
            
            else if (action.equals("DeleteEvent"))
            {
                String date = input.nextToken();
                String event = events.remove(date);

                if (event == null) {
                    System.out.println("DeleteEvent " + date + " success=noDateError");
                } else {
                    System.out.println("DeleteEvent " + date + " success");
                }
            }
            
            else if (action.equals("DisplayEventsBetweenDates"))
            {
                String startdate = input.nextToken();
                String enddate = input.nextToken();
                System.out.println("DisplayEventsBetweenDates " + startdate + " " + enddate + " " + events.subMap(startdate, enddate));
            }
            
            else if (action.equals("DisplayEventsFromStartDate"))
            {
                String startdate = input.nextToken();
                System.out.println("DisplayEventsFromStartDate " + startdate + " " + events.subMap(startdate, "9999"));
            }
            
            else if (action.equals("DisplayEventsToEndDate"))
            {
                String enddate = input.nextToken();
                System.out.println("DisplayEventsToEndDate " + enddate + " " + events.subMap("0000", enddate));
            }
            
            else if (action.equals("DisplayAllEvents"))
            {
                System.out.println("DisplayAllEvents " + events.allEvents());
            }
            
            else if (action.equals("PrintSkipList"))
            {
                System.out.println("PrintSkipList");
                events.printSkipList();
            }
        }
    }
}

