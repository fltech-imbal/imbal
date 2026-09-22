import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW1
{
    static class Representative
    {
        String name;

        Representative(String name)
        {
            this.name = name;
        }

        public String toString()
        {
            return name;
        }
    }

    static class Customer
    {
        String name;
        int requestTime;

        Customer(String name, int requestTime)
        {
            this.name = name;
            this.requestTime = requestTime;
        }

        public String toString()
        {
            return name;
        }
    }

    static class Chat
    {
        Customer customer;
        Representative representative;

        Chat(Customer customer, Representative representative)
        {
            this.customer = customer;
            this.representative = representative;
        }
    }

    static Chat removeChat(
        SinglyLinkedList<Chat> activeChats,
        String customerName)
    {
        SinglyLinkedList<Chat> temp =
            new SinglyLinkedList<Chat>();

        Chat found = null;

        while (!activeChats.isEmpty())
        {
            Chat chat = activeChats.removeFirst();

            if (chat.customer.name.equals(customerName))
            {
                found = chat;
            }
            else
            {
                temp.addLast(chat);
            }
        }

        while (!temp.isEmpty())
        {
            activeChats.addLast(temp.removeFirst());
        }

        return found;
    }

    static Customer removeCustomer(
        SinglyLinkedList<Customer> customersOnHold,
        String customerName)
    {
        SinglyLinkedList<Customer> temp =
            new SinglyLinkedList<Customer>();

        Customer found = null;

        while (!customersOnHold.isEmpty())
        {
            Customer customer = customersOnHold.removeFirst();

            if (customer.name.equals(customerName))
            {
                found = customer;
            }
            else
            {
                temp.addLast(customer);
            }
        }

        while (!temp.isEmpty())
        {
            customersOnHold.addLast(temp.removeFirst());
        }

        return found;
    }

    static void printAvailableRepList(
        SinglyLinkedList<Representative> availableReps,
        int time)
    {
        System.out.printf("AvailableRepList %04d", time);

        SinglyLinkedList<Representative> temp =
            new SinglyLinkedList<Representative>();

        while (!availableReps.isEmpty())
        {
            Representative rep = availableReps.removeFirst();

            System.out.print(" " + rep.name);

            temp.addLast(rep);
        }

        while (!temp.isEmpty())
        {
            availableReps.addLast(temp.removeFirst());
        }

        System.out.println();
    }

    public static void main(String[] args)
        throws FileNotFoundException
    {
        SinglyLinkedList<Representative> availableReps =
            new SinglyLinkedList<Representative>();

        SinglyLinkedList<Customer> customersOnHold =
            new SinglyLinkedList<Customer>();

        SinglyLinkedList<Chat> activeChats =
            new SinglyLinkedList<Chat>();

        availableReps.addLast(
            new Representative("Alice"));

        availableReps.addLast(
            new Representative("Bob"));

        availableReps.addLast(
            new Representative("Carol"));

        availableReps.addLast(
            new Representative("David"));

        availableReps.addLast(
            new Representative("Emily"));

        int maxWaitTime = 0;

        Scanner input =
            new Scanner(new File(args[0]));

        while (input.hasNextLine())
        {
            Scanner line =
                new Scanner(input.nextLine());

            String event = line.next();

            if (event.equals("ChatRequest"))
            {
                int requestTime = line.nextInt();
                String customerName = line.next();
                String decision = line.next();

                System.out.printf(
                    "ChatRequest %04d %s %s%n",
                    requestTime,
                    customerName,
                    decision);

                Customer customer =
                    new Customer(customerName, requestTime);

                if (!availableReps.isEmpty())
                {
                    Representative rep =
                        availableReps.removeFirst();

                    activeChats.addLast(
                        new Chat(customer, rep));

                    System.out.printf(
                        "RepAssignment %s %s %04d%n",
                        customerName,
                        rep.name,
                        requestTime);
                }
                else if (decision.equals("wait"))
                {
                    customersOnHold.addLast(customer);

                    System.out.printf(
                        "PutOnHold %s %04d%n",
                        customerName,
                        requestTime);
                }
                else
                {
                    System.out.printf(
                        "TryLater %s %04d%n",
                        customerName,
                        requestTime);
                }
            }

            else if (event.equals("ChatEnded"))
            {
                String customerName = line.next();
                String repName = line.next();
                int endTime = line.nextInt();

                Chat chat =
                    removeChat(activeChats, customerName);

                Representative rep;

                if (chat != null)
                {
                    rep = chat.representative;
                }
                else
                {
                    rep = new Representative(repName);
                }

                System.out.printf(
                    "ChatEnded %s %s %04d%n",
                    customerName,
                    repName,
                    endTime);

                if (!customersOnHold.isEmpty())
                {
                    Customer customer =
                        customersOnHold.removeFirst();

                    activeChats.addLast(
                        new Chat(customer, rep));

                    int waitTime =
                        endTime - customer.requestTime;

                    if (waitTime > maxWaitTime)
                    {
                        maxWaitTime = waitTime;
                    }

                    System.out.printf(
                        "RepAssignment %s %s %04d%n",
                        customer.name,
                        rep.name,
                        endTime);
                }
                else
                {
                    availableReps.addLast(rep);
                }
            }

            else if (event.equals("QuitOnHold"))
            {
                int quitTime = line.nextInt();
                String customerName = line.next();

                Customer customer =
                    removeCustomer(
                        customersOnHold,
                        customerName);

                System.out.printf(
                    "QuitOnHold %04d %s%n",
                    quitTime,
                    customerName);

                if (customer != null)
                {
                    int waitTime =
                        quitTime - customer.requestTime;

                    if (waitTime > maxWaitTime)
                    {
                        maxWaitTime = waitTime;
                    }
                }
            }

            else if (event.equals("PrintAvailableRepList"))
            {
                int printTime = line.nextInt();

                printAvailableRepList(
                    availableReps,
                    printTime);
            }

            else if (event.equals("PrintMaxWaitTime"))
            {
                int printTime = line.nextInt();

                System.out.printf(
                    "MaxWaitTime %04d %04d%n",
                    printTime,
                    maxWaitTime);
            }

            line.close();
        }

        input.close();
    }
}
