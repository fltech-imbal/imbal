/*

  Author: Matthew Gomez
  Email: matthewgomez2025@fit.edu
  Course: CSE 2010
  Section:
  Description of this file: Simulates an online retailer's chat support
  system. Chat requests, customers going on hold or quitting hold, chats
  ending, and status queries are read from an input file (one event per
  line, in time order) and processed against three singly linked lists:
  the representatives currently available, the customers currently on
  hold, and the chat sessions currently in progress. Each resulting
  event is printed to standard output in the format required by the
  assignment.

 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.StringTokenizer;

public class HW1
{
    /*
      Description of each method, including parameters
    */

    /**
     * Holds one customer who is currently on hold: their name and the
     * time (as it appeared in the input file) at which they originally
     * requested a chat. The original request time is kept both as a
     * String (so it can be echoed back exactly, preserving any leading
     * zeros from HHMM format) and as an int (so wait time can be
     * computed by subtraction).
     */
    private static class HoldCustomer
    {
        String name;
        String requestTimeStr;
        int requestTime;

        HoldCustomer(String name, String requestTimeStr, int requestTime)
        {
            this.name = name;
            this.requestTimeStr = requestTimeStr;
            this.requestTime = requestTime;
        }

        // Two HoldCustomer objects are treated as equal when they share
        // the same customer name. This lets SinglyLinkedList.remove(E)
        // find and remove a specific waiting customer by name alone,
        // which is all we know from a QuitOnHold line.
        public boolean equals(Object o)
        {
            if (!(o instanceof HoldCustomer)) return false;
            return this.name.equals(((HoldCustomer) o).name);
        }
    }

    /**
     * Holds one chat session currently in progress: which customer is
     * talking to which representative.
     */
    private static class ChatSession
    {
        String customer;
        String rep;

        ChatSession(String customer, String rep)
        {
            this.customer = customer;
            this.rep = rep;
        }

        // Two ChatSession objects are treated as equal when they involve
        // the same customer and the same representative, so the correct
        // session can be located and removed when its chat ends.
        public boolean equals(Object o)
        {
            if (!(o instanceof ChatSession)) return false;
            ChatSession other = (ChatSession) o;
            return this.customer.equals(other.customer) && this.rep.equals(other.rep);
        }
    }

    /**
     * Reads the simulation input file named in args[0], processes each
     * line in order, and prints the resulting events to standard output.
     *
     * @param args  command-line arguments; args[0] must be the path to
     *              the input file containing the chat request events
     */
    public static void main(String[] args)
    {
        /* description of variables */

        // availableReps: representatives who are currently free to take
        // a new chat, ordered from whoever has been available longest
        // (the front) to whoever became available most recently (the back)
        SinglyLinkedList<String> availableReps = new SinglyLinkedList<>();

        // onHold: customers currently on hold, ordered from whoever has
        // been waiting longest (the front) to whoever was put on hold
        // most recently (the back)
        SinglyLinkedList<HoldCustomer> onHold = new SinglyLinkedList<>();

        // chatSessions: chats currently in progress
        SinglyLinkedList<ChatSession> chatSessions = new SinglyLinkedList<>();

        // maxWaitTime: the largest wait time seen so far among customers
        // whose wait time has been determined, either by being assigned
        // a representative or by quitting while on hold
        int maxWaitTime = 0;

        // The 5 representatives are initially available in this order
        availableReps.addLast("Alice");
        availableReps.addLast("Bob");
        availableReps.addLast("Carol");
        availableReps.addLast("David");
        availableReps.addLast("Emily");

        if (args.length < 1)
        {
            System.out.println("Usage: java HW1 <inputFile>");
            return;
        }

        /* description of block: open the input file named on the
           command line, so each line of it can be read and processed
           one at a time below */
        try
        {
            BufferedReader reader = new BufferedReader(new FileReader(args[0]));
            String line;

            /* description of block: main simulation loop -- read one
               request per line, figure out which of the five request
               types it is, update whichever of the three linked lists
               are affected, and print every event the assignment asks
               for (the request itself, when applicable, followed
               immediately by whatever event it triggers) */
            while ((line = reader.readLine()) != null)
            {
                line = line.trim();
                if (line.isEmpty()) continue;

                StringTokenizer st = new StringTokenizer(line);
                String type = st.nextToken();

                if (type.equals("ChatRequest"))
                {
                    /* description of block: a customer is requesting a
                       chat. Echo the request, then either assign the
                       longest-available representative immediately, put
                       the customer on hold, or record that they will
                       try again later, depending on whether a
                       representative is currently free */
                    String requestTimeStr = st.nextToken();
                    int requestTime = Integer.parseInt(requestTimeStr);
                    String customer = st.nextToken();
                    String waitOrLater = st.nextToken();

                    System.out.println("ChatRequest " + requestTimeStr + " " + customer + " " + waitOrLater);

                    if (!availableReps.isEmpty())
                    {
                        String rep = availableReps.removeFirst();
                        chatSessions.addLast(new ChatSession(customer, rep));
                        System.out.println("RepAssignment " + customer + " " + rep + " " + requestTimeStr);
                    }
                    else if (waitOrLater.equals("wait"))
                    {
                        onHold.addLast(new HoldCustomer(customer, requestTimeStr, requestTime));
                        System.out.println("PutOnHold " + customer + " " + requestTimeStr);
                    }
                    else
                    {
                        System.out.println("TryLater " + customer + " " + requestTimeStr);
                    }
                }
                else if (type.equals("QuitOnHold"))
                {
                    /* description of block: a customer who was on hold
                       gives up waiting. Echo the event, remove that
                       customer from the hold list wherever they are in
                       it, and use their original request time to update
                       the running maximum wait time */
                    String quitTimeStr = st.nextToken();
                    int quitTime = Integer.parseInt(quitTimeStr);
                    String customer = st.nextToken();

                    System.out.println("QuitOnHold " + quitTimeStr + " " + customer);

                    int foundRequestTime = -1;
                    for (HoldCustomer hc : onHold)
                    {
                        if (hc.name.equals(customer))
                        {
                            foundRequestTime = hc.requestTime;
                            break;
                        }
                    }
                    onHold.remove(new HoldCustomer(customer, null, 0));

                    if (foundRequestTime >= 0)
                        maxWaitTime = Math.max(maxWaitTime, quitTime - foundRequestTime);
                }
                else if (type.equals("ChatEnded"))
                {
                    /* description of block: a chat session finishes.
                       Echo the event, remove the session, and return the
                       representative to the back of the available list.
                       If anyone is on hold, immediately assign the
                       longest-available representative to the
                       longest-waiting customer, since a rep just freed
                       up */
                    String customer = st.nextToken();
                    String rep = st.nextToken();
                    String endTimeStr = st.nextToken();
                    int endTime = Integer.parseInt(endTimeStr);

                    System.out.println("ChatEnded " + customer + " " + rep + " " + endTimeStr);

                    chatSessions.remove(new ChatSession(customer, rep));
                    availableReps.addLast(rep);

                    if (!onHold.isEmpty())
                    {
                        HoldCustomer nextCustomer = onHold.removeFirst();
                        String freeRep = availableReps.removeFirst();
                        chatSessions.addLast(new ChatSession(nextCustomer.name, freeRep));
                        maxWaitTime = Math.max(maxWaitTime, endTime - nextCustomer.requestTime);
                        System.out.println("RepAssignment " + nextCustomer.name + " " + freeRep + " " + endTimeStr);
                    }
                }
                else if (type.equals("PrintAvailableRepList"))
                {
                    /* description of block: print every representative
                       currently available, in order from longest
                       available to most recently freed */
                    String printTimeStr = st.nextToken();
                    StringBuilder sb = new StringBuilder("AvailableRepList " + printTimeStr);
                    for (String rep : availableReps)
                        sb.append(" ").append(rep);
                    System.out.println(sb.toString());
                }
                else if (type.equals("PrintMaxWaitTime"))
                {
                    /* description of block: print the largest wait time
                       observed so far across all customers whose wait
                       time has been determined */
                    String printTimeStr = st.nextToken();
                    System.out.println("MaxWaitTime " + printTimeStr + " " + maxWaitTime);
                }
            }

            reader.close();
        }
        catch (IOException ex)
        {
            System.out.println("Error reading input file: " + ex.getMessage());
        }
    }
}