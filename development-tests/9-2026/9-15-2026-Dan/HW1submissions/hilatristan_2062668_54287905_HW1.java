/*
  Author: Tristan Hila
  Email: thila2025@fit.edu
  Course: 2010
  Section: 04
  Description of this file:
  simulates the customer service chat system using singly linked lists
*/

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Scanner;

public class HW1 {
    // separate lists for available reps waiting customers and active chats
    private static final SinglyLinkedList<String> available = new SinglyLinkedList<String>();
    private static final SinglyLinkedList<Hold> onHold = new SinglyLinkedList<Hold>();
    private static final SinglyLinkedList<Chat> chats = new SinglyLinkedList<Chat>();
    private static int maxWaitMinutes;

    // stores a waiting customer and the original request time
    private static class Hold {
        final String customer; final int requestTime;
        Hold(String customer, int requestTime) { this.customer = customer; this.requestTime = requestTime; }
    }
    // stores the customer and representative in an active chat
    private static class Chat {
        final String customer; final String representative;
        Chat(String customer, String representative) { this.customer = customer; this.representative = representative; }
    }
    // stores a command read from the input file
    private static class InputEvent {
        final String line, kind; final int time, inputOrder;
        InputEvent(String line, String kind, int time, int inputOrder) {
            this.line = line; this.kind = kind; this.time = time; this.inputOrder = inputOrder;
        }
    }
    // stores an output line and its required ordering information
    private static class OutputEvent {
        final String text; final int priority, order;
        OutputEvent(String text, int priority, int order) { this.text = text; this.priority = priority; this.order = order; }
    }

    // initializes the representatives and processes each timestamp group
    public static void main(String[] args) throws FileNotFoundException {
        // require one command-line argument containing the input file name
        if (args.length != 1) {
            System.err.println("Usage: java HW1 input-file");
            return;
        }
        // representatives begin available in the required order
        available.addLast("Alice"); available.addLast("Bob"); available.addLast("Carol");
        available.addLast("David"); available.addLast("Emily");

        List<InputEvent> events = readEvents(args[0]);
        int start = 0;
        // process all events sharing a timestamp as one group
        while (start < events.size()) {
            int end = start + 1;
            while (end < events.size() && events.get(end).time == events.get(start).time) end++;
            processTimeGroup(events.subList(start, end));
            start = end;
        }
    }

    // reads all nonempty input lines and records their timestamps
    private static List<InputEvent> readEvents(String fileName) throws FileNotFoundException {
        List<InputEvent> events = new ArrayList<InputEvent>();
        Scanner input = new Scanner(new File(fileName));
        int order = 0;
        while (input.hasNextLine()) {
            String line = input.nextLine().trim();
            if (line.length() == 0) continue;
            String[] fields = line.split("\\s+");
            int time = fields[0].equals("ChatEnded") ? Integer.parseInt(fields[3]) : Integer.parseInt(fields[1]);
            events.add(new InputEvent(line, fields[0], time, order++));
        }
        input.close();
        return events;
    }

    // processes and prints one timestamp group in the required order
    private static void processTimeGroup(List<InputEvent> group) {
        List<InputEvent> ordered = new ArrayList<InputEvent>(group);
        Collections.sort(ordered, new Comparator<InputEvent>() {
            public int compare(InputEvent a, InputEvent b) {
                int d = inputPriority(a.kind) - inputPriority(b.kind);
                return d != 0 ? d : a.inputOrder - b.inputOrder;
            }
        });
        List<OutputEvent> output = new ArrayList<OutputEvent>();
        int sequence = 0;
        for (InputEvent event : ordered) sequence = process(event, output, sequence);
        Collections.sort(output, new Comparator<OutputEvent>() {
            public int compare(OutputEvent a, OutputEvent b) {
                int d = a.priority - b.priority;
                return d != 0 ? d : a.order - b.order;
            }
        });
        for (OutputEvent event : output) System.out.println(event.text);
    }

    // updates the lists and creates output for one input event
    private static int process(InputEvent event, List<OutputEvent> output, int sequence) {
        String[] f = event.line.split("\\s+");
        if (event.kind.equals("ChatRequest")) {
            String time = f[1], customer = f[2];
            add(output, event.line, "ChatRequest", sequence++);
            if (!available.isEmpty()) {
                String rep = available.removeFirst();
                chats.addLast(new Chat(customer, rep));
                add(output, "RepAssignment " + customer + " " + rep + " " + time, "RepAssignment", sequence++);
            } else if (f[3].equals("wait")) {
                onHold.addLast(new Hold(customer, event.time));
                add(output, "PutOnHold " + customer + " " + time, "PutOnHold", sequence++);
            } else add(output, "TryLater " + customer + " " + time, "TryLater", sequence++);
        } else if (event.kind.equals("ChatEnded")) {
            String customer = f[1], rep = f[2], time = f[3];
            removeChat(customer, rep);
            add(output, event.line, "ChatEnded", sequence++);
            if (!onHold.isEmpty()) {
                Hold next = onHold.removeFirst();
                chats.addLast(new Chat(next.customer, rep));
                updateMaxWait(next.requestTime, event.time);
                add(output, "RepAssignment " + next.customer + " " + rep + " " + time, "RepAssignment", sequence++);
            } else available.addLast(rep);
        } else if (event.kind.equals("QuitOnHold")) {
            Hold quitter = removeHold(f[2]);
            if (quitter != null) updateMaxWait(quitter.requestTime, event.time);
            add(output, event.line, "QuitOnHold", sequence++);
        } else if (event.kind.equals("PrintAvailableRepList")) {
            StringBuilder line = new StringBuilder("AvailableRepList ").append(f[1]);
            for (String rep : available) line.append(' ').append(rep);
            add(output, line.toString(), "AvailableRepList", sequence++);
        } else if (event.kind.equals("PrintMaxWaitTime")) {
            add(output, "MaxWaitTime " + f[1] + " " + formatDuration(maxWaitMinutes), "MaxWaitTime", sequence++);
        }
        return sequence;
    }

    // removes and returns the named customer from the hold list
    private static Hold removeHold(String customer) {
        for (Hold hold : onHold) if (hold.customer.equals(customer)) { onHold.remove(hold); return hold; }
        return null;
    }
    // removes a completed session from the active chat list
    private static void removeChat(String customer, String rep) {
        for (Chat chat : chats) if (chat.customer.equals(customer) && chat.representative.equals(rep)) { chats.remove(chat); return; }
    }
    // updates the longest completed wait time
    private static void updateMaxWait(int start, int end) {
        int wait = toMinutes(end) - toMinutes(start);
        if (wait > maxWaitMinutes) maxWaitMinutes = wait;
    }
    // converts an HHMM time to total minutes
    private static int toMinutes(int hhmm) { return (hhmm / 100) * 60 + hhmm % 100; }
    // formats a minute duration as HHMM
    private static String formatDuration(int minutes) { return String.format("%02d%02d", minutes / 60, minutes % 60); }
    // adds a line to the pending output list
    private static void add(List<OutputEvent> out, String text, String kind, int order) {
        out.add(new OutputEvent(text, outputPriority(kind), order));
    }
    // returns the processing priority of an input command
    private static int inputPriority(String kind) {
        if (kind.equals("ChatEnded")) return 0;
        if (kind.equals("ChatRequest")) return 1;
        if (kind.equals("QuitOnHold")) return 2;
        if (kind.equals("PrintAvailableRepList")) return 3;
        return 4;
    }
    // returns the required printing priority of an output event
    private static int outputPriority(String kind) {
        if (kind.equals("ChatEnded")) return 0;
        if (kind.equals("ChatRequest")) return 1;
        if (kind.equals("RepAssignment")) return 2;
        if (kind.equals("PutOnHold")) return 3;
        if (kind.equals("TryLater")) return 4;
        if (kind.equals("QuitOnHold")) return 5;
        if (kind.equals("AvailableRepList")) return 6;
        return 7;
    }
}
