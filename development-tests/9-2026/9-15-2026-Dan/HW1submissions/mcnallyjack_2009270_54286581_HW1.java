import java.util.Scanner;

public class HW1
{
    /*
     * RepNode stores one representative and a reference to
     * the next representative. This creates the singly linked
     * list used for representatives who are available.
     */
    static class RepNode
    {
        String name;
        RepNode next;

        RepNode(String name)
        {
            this.name = name;
            next = null;
        }
    }

    /*
     * HoldNode stores one customer who is waiting for a
     * representative. It stores the customer name, request
     * time, and reference to the next waiting customer.
     */
    static class HoldNode
    {
        String customer;
        int requestTime;
        HoldNode next;

        HoldNode(String customer, int requestTime)
        {
            this.customer = customer;
            this.requestTime = requestTime;
            next = null;
        }
    }

    /*
     * SessionNode stores one active chat session. It stores
     * the customer name, representative name, request time,
     * and reference to the next active chat session.
     */
    static class SessionNode
    {
        String customer;
        String rep;
        int requestTime;
        SessionNode next;

        SessionNode(String customer, String rep, int requestTime)
        {
            this.customer = customer;
            this.rep = rep;
            this.requestTime = requestTime;
            next = null;
        }
    }

    static RepNode availableHead = null;
    static HoldNode holdHead = null;
    static SessionNode sessionHead = null;

    static int maxWaitTime = 0;

    /*
     * Adds a representative to the end of the available
     * representative list. The parameter is the representative's
     * name.
     */
    public static void addRepresentative(String name)
    {
        RepNode newNode = new RepNode(name);

        if (availableHead == null)
        {
            availableHead = newNode;
            return;
        }

        RepNode current = availableHead;

        while (current.next != null)
        {
            current = current.next;
        }

        current.next = newNode;
    }

    /*
     * Removes the first representative from the available
     * representative list. The method returns the name of
     * the representative that was removed.
     */
    public static String removeRepresentative()
    {
        if (availableHead == null)
        {
            return null;
        }

        String name = availableHead.name;
        availableHead = availableHead.next;

        return name;
    }

    /*
     * Adds a customer to the end of the hold list. The
     * parameters are the customer's name and original
     * request time.
     */
    public static void addCustomerOnHold(
        String customer, int requestTime)
    {
        HoldNode newNode =
            new HoldNode(customer, requestTime);

        if (holdHead == null)
        {
            holdHead = newNode;
            return;
        }

        HoldNode current = holdHead;

        while (current.next != null)
        {
            current = current.next;
        }

        current.next = newNode;
    }

    /*
     * Removes a customer from the hold list. The parameter
     * is the customer's name. The method returns the node
     * that was removed.
     */
    public static HoldNode removeCustomerOnHold(
        String customer)
    {
        HoldNode current = holdHead;
        HoldNode previous = null;

        while (current != null)
        {
            if (current.customer.equals(customer))
            {
                if (previous == null)
                {
                    holdHead = current.next;
                }
                else
                {
                    previous.next = current.next;
                }

                current.next = null;
                return current;
            }

            previous = current;
            current = current.next;
        }

        return null;
    }

    /*
     * Adds an active chat session to the session list. The
     * parameters are the customer name, representative name,
     * and original request time.
     */
    public static void addSession(
        String customer, String rep, int requestTime)
    {
        SessionNode newNode =
            new SessionNode(customer, rep, requestTime);

        if (sessionHead == null)
        {
            sessionHead = newNode;
            return;
        }

        SessionNode current = sessionHead;

        while (current.next != null)
        {
            current = current.next;
        }

        current.next = newNode;
    }

    /*
     * Removes an active chat session for a customer. The
     * parameter is the customer's name. The method returns
     * the removed session node.
     */
    public static SessionNode removeSession(String customer)
    {
        SessionNode current = sessionHead;
        SessionNode previous = null;

        while (current != null)
        {
            if (current.customer.equals(customer))
            {
                if (previous == null)
                {
                    sessionHead = current.next;
                }
                else
                {
                    previous.next = current.next;
                }

                current.next = null;
                return current;
            }

            previous = current;
            current = current.next;
        }

        return null;
    }

    /*
     * Converts an HHMM time into the total number of minutes.
     * For example, 0800 becomes 480 minutes. The parameter
     * is the time in HHMM format.
     */
    public static int convertToMinutes(int time)
    {
        int hours = time / 100;
        int minutes = time % 100;

        return hours * 60 + minutes;
    }

    /*
     * Converts a number of minutes into HHMM format. The
     * parameter is the number of minutes. The method returns
     * a four-digit String representing the time.
     */
    public static String convertToHHMM(int minutes)
    {
        int hours = minutes / 60;
        int mins = minutes % 60;

        return String.format("%02d%02d", hours, mins);
    }

    /*
     * Updates the maximum wait time. The parameters are the
     * original request time and the time when the customer
     * was assigned or quit waiting.
     */
    public static void updateMaxWait(
        int requestTime, int finishTime)
    {
        int waitTime =
            convertToMinutes(finishTime)
            - convertToMinutes(requestTime);

        if (waitTime > maxWaitTime)
        {
            maxWaitTime = waitTime;
        }
    }

    /*
     * Processes a ChatRequest. If a representative is
     * available, the customer is immediately assigned.
     * Otherwise, the customer waits or tries later.
     */
    public static void processChatRequest(
        int requestTime,
        String customer,
        String decision)
    {
        System.out.println(
            "ChatRequest " +
            convertToHHMM(requestTime) +
            " " +
            customer +
            " " +
            decision);

        if (availableHead != null)
        {
            String rep = removeRepresentative();

            addSession(
                customer,
                rep,
                requestTime);

            System.out.println(
                "RepAssignment " +
                customer +
                " " +
                rep +
                " " +
                convertToHHMM(requestTime));
        }
        else
        {
            if (decision.equals("wait"))
            {
                addCustomerOnHold(
                    customer,
                    requestTime);

                System.out.println(
                    "PutOnHold " +
                    customer +
                    " " +
                    convertToHHMM(requestTime));
            }
            else
            {
                System.out.println(
                    "TryLater " +
                    customer +
                    " " +
                    convertToHHMM(requestTime));
            }
        }
    }

    /*
     * Processes a ChatEnded event. The representative becomes
     * available and is added to the end of the representative
     * list. The first customer on hold receives that representative.
     */
    public static void processChatEnded(
        String customer,
        String rep,
        int endTime)
    {
        System.out.println(
            "ChatEnded " +
            customer +
            " " +
            rep +
            " " +
            convertToHHMM(endTime));

        SessionNode session =
            removeSession(customer);

        if (session != null)
        {
            updateMaxWait(
                session.requestTime,
                endTime);
        }

        addRepresentative(rep);

        if (holdHead != null)
        {
            HoldNode waitingCustomer = holdHead;

            holdHead = holdHead.next;
            waitingCustomer.next = null;

            addSession(
                waitingCustomer.customer,
                rep,
                waitingCustomer.requestTime);

            System.out.println(
                "RepAssignment " +
                waitingCustomer.customer +
                " " +
                rep +
                " " +
                convertToHHMM(endTime));

            updateMaxWait(
                waitingCustomer.requestTime,
                endTime);
        }
    }

    /*
     * Processes a customer quitting while on hold. The
     * customer is removed from the hold list and the wait
     * time is used to update the maximum wait time.
     */
    public static void processQuitOnHold(
        int quitTime,
        String customer)
    {
        HoldNode customerNode =
            removeCustomerOnHold(customer);

        if (customerNode != null)
        {
            updateMaxWait(
                customerNode.requestTime,
                quitTime);
        }

        System.out.println(
            "QuitOnHold " +
            convertToHHMM(quitTime) +
            " " +
            customer);
    }

    /*
     * Prints the available representative list. The parameter
     * is the time when the list is being printed. The list is
     * printed from the first node through the last node.
     */
    public static void printAvailableRepList(int printTime)
    {
        System.out.print(
            "AvailableRepList " +
            convertToHHMM(printTime));

        RepNode current = availableHead;

        while (current != null)
        {
            System.out.print(
                " " + current.name);

            current = current.next;
        }

        System.out.println();
    }

    /*
     * Prints the maximum wait time recorded so far. The
     * parameter is the time when the maximum wait time
     * is requested.
     */
    public static void printMaxWaitTime(int printTime)
    {
        System.out.println(
            "MaxWaitTime " +
            convertToHHMM(printTime) +
            " " +
            convertToHHMM(maxWaitTime));
    }

    public static void main(String[] args)
    {
        /*
         * The Scanner reads one complete event from the keyboard.
         * The program stays running so that another event can be
         * entered after the previous event has been processed.
         */
        Scanner input = new Scanner(System.in);

        String event;
        int time;
        String customer;
        String rep;
        String decision;

        /*
         * Create the five representatives and place them in
         * the available representative list. The required
         * initial order is Alice, Bob, Carol, David, Emily.
         */
        addRepresentative("Alice");
        addRepresentative("Bob");
        addRepresentative("Carol");
        addRepresentative("David");
        addRepresentative("Emily");

        /*
         * Continuously read one event from the keyboard.
         * Each time the user presses Enter, the event is
         * processed immediately and its output is displayed.
         */
        while (input.hasNext())
        {
            event = input.next();

            /*
             * A ChatRequest contains the request time,
             * customer name, and either wait or later.
             * The event is immediately processed.
             */
            if (event.equals("ChatRequest"))
            {
                time = input.nextInt();
                customer = input.next();
                decision = input.next();

                processChatRequest(
                    time,
                    customer,
                    decision);
            }

            /*
             * A ChatEnded event contains the customer,
             * representative, and ending time. The chat
             * is ended and the representative is released.
             */
            else if (event.equals("ChatEnded"))
            {
                customer = input.next();
                rep = input.next();
                time = input.nextInt();

                processChatEnded(
                    customer,
                    rep,
                    time);
            }

            /*
             * A QuitOnHold event contains the quit time and
             * customer name. The customer is removed from
             * the hold list and the wait time is calculated.
             */
            else if (event.equals("QuitOnHold"))
            {
                time = input.nextInt();
                customer = input.next();

                processQuitOnHold(
                    time,
                    customer);
            }

            /*
             * PrintAvailableRepList contains only the time.
             * The current available representative list is
             * printed immediately after reading the event.
             */
            else if (event.equals("PrintAvailableRepList"))
            {
                time = input.nextInt();

                printAvailableRepList(time);
            }

            /*
             * PrintMaxWaitTime contains only the print time.
             * The largest wait time recorded so far is printed
             * immediately after reading the event.
             */
            else if (event.equals("PrintMaxWaitTime"))
            {
                time = input.nextInt();

                printMaxWaitTime(time);
            }
        }

        input.close();
    }
}