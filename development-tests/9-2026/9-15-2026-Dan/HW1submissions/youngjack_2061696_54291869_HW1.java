/*

* Author: Jack Young
* Email: jack2025@fit.edu
* Course: CSE2010: Algorithms & Data Struct
* Section: 1
*
* Description of this file:
 A program that simulates an online customer service chat system.
  */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Comparator;

public class HW1
{

private static class Representative
{
String name;

    public Representative(String name)
    {
        this.name = name;
    }

    public String toString()
    {
        return name;
    }
}


private static class Customer
{
    String name;
    int requestTime;

    public Customer(String name, int requestTime)
    {
        this.name = name;
        this.requestTime = requestTime;
    }

    public String toString()
    {
        return name;
    }
}


private static class ChatSession
{
    Customer customer;
    Representative representative;

    public ChatSession(Customer customer,
                       Representative representative)
    {
        this.customer = customer;
        this.representative = representative;
    }
}

/*
 * Represents one input event.
 */
private static class Event
{
    String type;
    int time;
    String customer;
    String representative;
    String decision;

    public Event(String type, int time)
    {
        this.type = type;
        this.time = time;
    }
}

/*
 * Represents a representative assignment that will be printed
 * after all ChatRequest events at the same time since 
 * the assignment event has a lower priority than ChatRequest.
 */
private static class Assignment
{
    Customer customer;
    Representative representative;
    int time;

    public Assignment(Customer customer,
                      Representative representative,
                      int time)
    {
        this.customer = customer;
        this.representative = representative;
        this.time = time;
    }
}

/*
 * Converts an HHMM integer into the number of minutes since midnight.
 */
private static int toMinutes(int time)
{
    int hours = time / 100;
    int minutes = time % 100;

    return hours * 60 + minutes;
}

/*
 * Converts a time into HHMM format.
 */
private static String formatTime(int time)
{
    return String.format("%04d", time);
}



/*
 * Removes a particular customer from the on-hold list.
 * The remaining customers stay in exactly the same order.
 */
private static Customer removeCustomerOnHold(
        SinglyLinkedList<Customer> list, String name)
{
    SinglyLinkedList<Customer> temporary =
            new SinglyLinkedList<Customer>();

    Customer removed = null;

    while (!list.isEmpty())
    {
        Customer customer = list.removeFirst();

        if (removed == null && customer.name.equals(name))
        {
            removed = customer;
        }
        else
        {
            temporary.addLast(customer);
        }
    }

    while (!temporary.isEmpty())
    {
        list.addLast(temporary.removeFirst());
    }

    return removed;
}

/*
 * Removes the active chat session belonging to a customer.
 * The list is temporarily rebuilt since the supplied singly linked 
 * list only directly removes its first element.
 */
private static ChatSession removeSession(
        SinglyLinkedList<ChatSession> list, String customerName)
{
    SinglyLinkedList<ChatSession> temporary =
            new SinglyLinkedList<ChatSession>();

    ChatSession removed = null;
    while (!list.isEmpty())
    {
        ChatSession session = list.removeFirst();

        if (removed == null &&
            session.customer.name.equals(customerName))
        {
            removed = session;
        }
        else
        {
            temporary.addLast(session);
        }
    }

    while (!temporary.isEmpty())
    {
        list.addLast(temporary.removeFirst());
    }

    return removed;
}



/*
 * Returns the priority of an input event. The assignment requires
 * ChatEnded to occur first at a timestamp, followed by ChatRequest,
 * QuitOnHold, PrintAvailableRepList, and PrintMaxWaitTime.
 */
private static int eventPriority(String type)
{
    if (type.equals("ChatEnded"))
        return 0;

    if (type.equals("ChatRequest"))
        return 1;

    if (type.equals("QuitOnHold"))
        return 2;

    if (type.equals("PrintAvailableRepList"))
        return 3;

    if (type.equals("PrintMaxWaitTime"))
        return 4;

    return 5;
}

/*
 * Reads the input file and creates an Event object for each line.
 * The input is stored first so that events having the same timestamp
 * can be processed according to the required priority instead of
 * depending entirely on the order in which they appear in the file.
 */
private static ArrayList<Event> readEvents(String fileName)
        throws Exception
{
    ArrayList<Event> events = new ArrayList<Event>();

    BufferedReader input =
            new BufferedReader(new FileReader(fileName));

    String line;

    while ((line = input.readLine()) != null)
    {
        line = line.trim();

        if (line.length() == 0)
            continue;

        String[] parts = line.split("\\s+");

Event event;

if (parts[0].equals("ChatRequest"))
{
    event = new Event(parts[0],
                      Integer.parseInt(parts[1]));

    event.customer = parts[2];
    event.decision = parts[3];
}
else if (parts[0].equals("QuitOnHold"))
{
    event = new Event(parts[0],
                      Integer.parseInt(parts[1]));

    event.customer = parts[2];
}
else if (parts[0].equals("ChatEnded"))
{
    event = new Event(parts[0],
                      Integer.parseInt(parts[3]));

    event.customer = parts[1];
    event.representative = parts[2];
}
else if (parts[0].equals("PrintAvailableRepList"))
{
    event = new Event(parts[0],
                      Integer.parseInt(parts[1]));
}
else if (parts[0].equals("PrintMaxWaitTime"))
{
    event = new Event(parts[0],
                      Integer.parseInt(parts[1]));
}
else
{
    continue;
}

events.add(event);
    }

    input.close();

    events.sort(new Comparator<Event>()
    {
        public int compare(Event a, Event b)
        {
            if (a.time != b.time)
                return Integer.compare(a.time, b.time);

            return Integer.compare(eventPriority(a.type),
                                   eventPriority(b.type));
        }
    });

    return events;
}

/*
 * Main method.
 */
public static void main(String[] args)
{
    if (args.length < 1)
    {
        System.out.println("Please provide an input file name.");
        return;
    }

    try
    {
        ArrayList<Event> events = readEvents(args[0]);
        SinglyLinkedList<Representative> availableReps =
                new SinglyLinkedList<Representative>();

        SinglyLinkedList<Customer> customersOnHold =
                new SinglyLinkedList<Customer>();

        SinglyLinkedList<ChatSession> chatSessions =
                new SinglyLinkedList<ChatSession>();

        availableReps.addLast(new Representative("Alice"));
        availableReps.addLast(new Representative("Bob"));
        availableReps.addLast(new Representative("Carol"));
        availableReps.addLast(new Representative("David"));
        availableReps.addLast(new Representative("Emily"));


        int maxWaitTime = 0;

        int index = 0;
      
        while (index < events.size())
        {
            int currentTime = events.get(index).time;

            int endStart = index;

            while (index < events.size() &&
                   events.get(index).time == currentTime)
            {
                index++;
            }

            int endFinish = index;

            for (int i = endStart; i < endFinish; i++)
            {
                Event event = events.get(i);

                if (event.type.equals("ChatEnded"))
                {
                    ChatSession session =
                            removeSession(chatSessions,
                                          event.customer);

                    Representative rep;

                    if (session != null)
                        rep = session.representative;
                    else
                        rep = new Representative(event.representative);

                    availableReps.addLast(rep);

                    System.out.println(
                        "ChatEnded " +
                        event.customer + " " +
                        rep.name + " " +
                        formatTime(event.time));
                }
            }

            ArrayList<Assignment> assignments =
                    new ArrayList<Assignment>();

            ArrayList<Customer> holdDecisions =
                    new ArrayList<Customer>();

            ArrayList<Customer> laterDecisions =
                    new ArrayList<Customer>();

            for (int i = endStart; i < endFinish; i++)
            {
                Event event = events.get(i);

                if (event.type.equals("ChatRequest"))
                {
                    System.out.println(
                        "ChatRequest " +
                        formatTime(event.time) + " " +
                        event.customer + " " +
                        event.decision);

                    Customer customer =
                            new Customer(event.customer,
                                         event.time);

            
                    if (customersOnHold.isEmpty() &&
                        !availableReps.isEmpty())
                    {
                        Representative rep =
                                availableReps.removeFirst();

                        ChatSession session =
                                new ChatSession(customer, rep);

                        chatSessions.addLast(session);

                        assignments.add(
                            new Assignment(customer,
                                           rep,
                                           event.time));

                        int wait =
                            toMinutes(event.time) -
                            toMinutes(customer.requestTime);

                        if (wait > maxWaitTime)
                            maxWaitTime = wait;
                    }
                    else
                    {
                        /*
                         * There is no representative available for
                         * this customer. The customer's decision
                         * determines whether the customer enters
                         * the hold list or tries again later.
                         */
                        if (event.decision.equals("wait"))
                        {
                            customersOnHold.addLast(customer);
                            holdDecisions.add(customer);
                        }
                        else
                        {
                            laterDecisions.add(customer);
                        }
                    }
                }
            }

            while (!customersOnHold.isEmpty() &&
                   !availableReps.isEmpty())
            {
                Customer customer =
                        customersOnHold.removeFirst();

                Representative rep =
                        availableReps.removeFirst();

                ChatSession session =
                        new ChatSession(customer, rep);

                chatSessions.addLast(session);

                assignments.add(
                    new Assignment(customer,
                                   rep,
                                   currentTime));

                int wait =
                    toMinutes(currentTime) -
                    toMinutes(customer.requestTime);

                if (wait > maxWaitTime)
                    maxWaitTime = wait;
            }

            for (Assignment assignment : assignments)
            {
                System.out.println(
                    "RepAssignment " +
                    assignment.customer.name + " " +
                    assignment.representative.name + " " +
                    formatTime(assignment.time));
            }

            for (Customer customer : holdDecisions)
            {
                System.out.println(
                    "PutOnHold " +
                    customer.name + " " +
                    formatTime(currentTime));
            }

            for (Customer customer : laterDecisions)
            {
                System.out.println(
                    "TryLater " +
                    customer.name + " " +
                    formatTime(currentTime));
            }

            for (int i = endStart; i < endFinish; i++)
            {
                Event event = events.get(i);

                if (event.type.equals("QuitOnHold"))
                {
                    Customer customer =
                            removeCustomerOnHold(
                                customersOnHold,
                                event.customer);

                    System.out.println(
                        "QuitOnHold " +
                        formatTime(event.time) + " " +
                        event.customer);

                    if (customer != null)
                    {
                        int wait =
                            toMinutes(event.time) -
                            toMinutes(customer.requestTime);

                        if (wait > maxWaitTime)
                            maxWaitTime = wait;
                    }
                }
            }

            for (int i = endStart; i < endFinish; i++)
            {
                Event event = events.get(i);

                if (event.type.equals("PrintAvailableRepList"))
                {
                    System.out.print(
                        "AvailableRepList " +
                        formatTime(event.time));

                    SinglyLinkedList<Representative> temporary =
                            new SinglyLinkedList<Representative>();

                    while (!availableReps.isEmpty())
                    {
                        Representative rep =
                                availableReps.removeFirst();

                        System.out.print(" " + rep.name);

                        temporary.addLast(rep);
                    }

                    while (!temporary.isEmpty())
                    {
                        availableReps.addLast(
                            temporary.removeFirst());
                    }

                    System.out.println();
                }

                if (event.type.equals("PrintMaxWaitTime"))
                {
                    System.out.println(
                        "MaxWaitTime " +
                        formatTime(event.time) + " " +
                        maxWaitTime);
                }
            }
        }
    }
    catch (Exception e)
    {
        System.out.println("Error: " + e.getMessage());
    }
}



}
