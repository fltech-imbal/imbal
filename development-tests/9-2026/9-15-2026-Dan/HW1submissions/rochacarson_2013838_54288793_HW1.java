import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
//Carson Rocha for CSE2010 HW1
public class HW1 {

    // Basic node for the representative list
    static class RepNode {
        String name;
        RepNode next;

        RepNode(String name) {
            this.name = name;
            next = null;
        }
    }

    // List of repersentives that are available
    static class RepList {
        RepNode head;
        RepNode tail;

        // Adds or removes repersentaves 
        void add(String name) {
            RepNode newNode = new RepNode(name);

            if (head == null) {
                head = newNode;
                tail = newNode;
            } else {
                tail.next = newNode;
                tail = newNode;
            }
        }

      
        String remove() {
            if (head == null) {
                return null;
            }

            String name = head.name;
            head = head.next;

            if (head == null) {
                tail = null;
            }

            return name;
        }

        // Checks if there are any  repersentives available
        boolean empty() {
            return head == null;
        }

        // Prints all  repersentivess that are free
        void print() {
            RepNode current = head;

            while (current != null) {
                System.out.print(" " + current.name);
                current = current.next;
            }

            System.out.println();
        }
    }

    // Stores waiting customers
    static class CustomerNode {
        String name;
        int requestTime;
        CustomerNode next;

        CustomerNode(String name, int requestTime) {
            this.name = name;
            this.requestTime = requestTime;
            next = null;
        }
    }

    // List of waiting customrs
    static class CustomerList {
        CustomerNode head;
        CustomerNode tail;

        // Adds a customer to at end of  waiting list
        void add(String name, int requestTime) {
            CustomerNode newNode =
                new CustomerNode(name, requestTime);

            if (head == null) {
                head = newNode;
                tail = newNode;
            } else {
                tail.next = newNode;
                tail = newNode;
            }
        }

        // Removes  first customer in line
        CustomerNode removeFirst() {
            if (head == null) {
                return null;
            }

            CustomerNode temp = head;
            head = head.next;

            if (head == null) {
                tail = null;
            }

            return temp;
        }

        //  removes a specific customer
        CustomerNode remove(String name) {
            CustomerNode current = head;
            CustomerNode previous = null;

            while (current != null) {

                if (current.name.equals(name)) {

                    if (previous == null) {
                        head = current.next;
                    } else {
                        previous.next = current.next;
                    }

                    if (current == tail) {
                        tail = previous;
                    }

                    return current;
                }

                previous = current;
                current = current.next;
            }

            return null;
        }

        // Checks if anybody is waiting
        boolean empty() {
            return head == null;
        }
    }

    // Stores a live chat
    static class SessionNode {
        String customer;
        String rep;
        SessionNode next;

        SessionNode(String customer, String rep) {
            this.customer = customer;
            this.rep = rep;
            next = null;
        }
    }

    // List of chats that are live
    static class SessionList {
        SessionNode head;

        // Adds a new chat 
        void add(String customer, String rep) {
            SessionNode newNode =
                new SessionNode(customer, rep);

            if (head == null) {
                head = newNode;
            } else {
                SessionNode current = head;

                while (current.next != null) {
                    current = current.next;
                }

                current.next = newNode;
            }
        }

        // Removes a customer's active chat
        SessionNode remove(String customer) {
            SessionNode current = head;
            SessionNode previous = null;

            while (current != null) {

                if (current.customer.equals(customer)) {

                    if (previous == null) {
                        head = current.next;
                    } else {
                        previous.next = current.next;
                    }

                    return current;
                }

                previous = current;
                current = current.next;
            }

            return null;
        }
    }

    // Stores information about one event
    static class Event {
        String type;
        int time;
        String customer;
        String rep;
        String wait;

        Event next;

        Event(String type, int time) {
            this.type = type;
            this.time = time;
        }
    }

    // Keeps all the events in chronolgoical order
    static class EventList {
        Event head;

        // Adds an event in the correct time 
        void add(Event newEvent) {

            if (head == null) {
                head = newEvent;
                return;
            }

            Event current = head;
            Event previous = null;

            while (current != null &&
                   current.time <= newEvent.time) {

                previous = current;
                current = current.next;
            }

            if (previous == null) {
                newEvent.next = head;
                head = newEvent;
            } else {
                previous.next = newEvent;
                newEvent.next = current;
            }
        }

        // Removes  first event
        Event removeFirst() {
            if (head == null) {
                return null;
            }

            Event temp = head;
            head = head.next;

            return temp;
        }
    }

    // The three  lists thaT HOLD ALL THE INFORAMTION OF THE UNIVERSE!!!!
    static RepList reps = new RepList();
    static CustomerList waiting = new CustomerList();
    static SessionList chats = new SessionList();

    // Keeps track of the poor guy who waited the longest
    static int maxWait = 0;

    // super duper converter changes everything to minutes
    static int minutes(int time) {
        int hour = time / 100;
        int minute = time % 100;

        return hour * 60 + minute;
    }

    // changes the longest wait timne to the new guy who waited the longest
    static void updateWait(int requestTime, int endTime) {

        int wait = minutes(endTime) - minutes(requestTime);

        if (wait > maxWait) {
            maxWait = wait;
        }
    }

    // Reads the input file and cause im aweome throws a exception aka a error when you dont give it a file
    static EventList readFile(String filename)
        throws FileNotFoundException {

        EventList events = new EventList();

        Scanner input = new Scanner(new File(filename));

        while (input.hasNext()) {

            String type = input.next();

            // somebody wants to start a chat
            if (type.equals("ChatRequest")) {

                int time = input.nextInt();
                String customer = input.next();
                String wait = input.next();

                Event event = new Event(type, time);

                event.customer = customer;
                event.wait = wait;

                events.add(event);
            }

            // the poor guy got fed up with waiting and gave up
            else if (type.equals("QuitOnHold")) {

                int time = input.nextInt();
                String customer = input.next();

                Event event = new Event(type, time);

                event.customer = customer;

                events.add(event);
            }

            // the repersentive solved da issue so chat ends
            else if (type.equals("ChatEnded")) {

                String customer = input.next();
                String rep = input.next();
                int time = input.nextInt();

                Event event = new Event(type, time);

                event.customer = customer;
                event.rep = rep;

                events.add(event);
            }

            // whos free and not with a customer prints the reps 
            else if (type.equals("PrintAvailableRepList")) {

                int time = input.nextInt();

                Event event =
                    new Event(type, time);

                events.add(event);
            }

            // Prints the longest wait time so we know how much that poor guy is suffering 
            else if (type.equals("PrintMaxWaitTime")) {

                int time = input.nextInt();

                Event event =
                    new Event(type, time);

                events.add(event);
            }
        }

        input.close();

        return events;
    }

    public static void main(String[] args)
        throws FileNotFoundException {

        // Makes dang  sure an input file was provided
        if (args.length != 1) {
            System.out.println(
                "Please provide an input file."
            );
            return;
        }

        // sets the starting line up of repersentives
        reps.add("Alice");
        reps.add("Bob");
        reps.add("Carol");
        reps.add("David");
        reps.add("Emily");

        // Read  input file
        EventList events = readFile(args[0]);

        Event event;

        // Goes through all the events
        while ((event = events.removeFirst()) != null) {

            // Handles  new chat request
            if (event.type.equals("ChatRequest")) {

                System.out.println(
                    "ChatRequest " +
                    event.time + " " +
                    event.customer + " " +
                    event.wait
                );

                // If a rep is available, assign them so nobody is stuck forever
                if (!reps.empty()) {

                    String rep = reps.remove();

                    chats.add(
                        event.customer,
                        rep
                    );

                    System.out.println(
                        "RepAssignment " +
                        event.customer + " " +
                        rep + " " +
                        event.time
                    );

                } else {

                    // Customer wants to wait idk why but they do
                    if (event.wait.equals("wait")) {

                        waiting.add(
                            event.customer,
                            event.time
                        );

                        System.out.println(
                            "PutOnHold " +
                            event.customer + " " +
                            event.time
                        );

                    } else {

                        // Customer decides to try again later smart choice
                        System.out.println(
                            "TryLater " +
                            event.customer + " " +
                            event.time
                        );
                    }
                }
            }

            // Handles chat endings
            else if (event.type.equals("ChatEnded")) {

                System.out.println(
                    "ChatEnded " +
                    event.customer + " " +
                    event.rep + " " +
                    event.time
                );

                // Removes finished chat
                chats.remove(event.customer);

                // The rep is finished with last guy so ready for new person
                reps.add(event.rep);

                // Gives the rep to the next in line
                if (!waiting.empty()) {

                    CustomerNode customer =
                        waiting.removeFirst();

                    String rep = reps.remove();

                    updateWait(
                        customer.requestTime,
                        event.time
                    );

                    chats.add(
                        customer.name,
                        rep
                    );

                    System.out.println(
                        "RepAssignment " +
                        customer.name + " " +
                        rep + " " +
                        event.time
                    );
                }
            }

            // Handle someone giving up  on the waiting list
            else if (event.type.equals("QuitOnHold")) {

                CustomerNode customer =
                    waiting.remove(event.customer);

                if (customer != null) {

                    updateWait(
                        customer.requestTime,
                        event.time
                    );
                }

                System.out.println(
                    "QuitOnHold " +
                    event.time + " " +
                    event.customer
                );
            }

            // Print the current available rep list
            else if (
                event.type.equals(
                    "PrintAvailableRepList"
                )
            ) {

                System.out.print(
                    "AvailableRepList " +
                    event.time
                );

                reps.print();
            }

            // Print the longest wait so far
            else if (
                event.type.equals(
                    "PrintMaxWaitTime"
                )
            ) {

                System.out.println(
                    "MaxWaitTime " +
                    event.time + " " +
                    maxWait
                );
            }
        }
    }
}


