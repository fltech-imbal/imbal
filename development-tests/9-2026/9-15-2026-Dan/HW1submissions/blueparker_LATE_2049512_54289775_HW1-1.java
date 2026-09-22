/*

  Author: Parker Blue
  Email: pblue2025@my.fit.edu
  Course: CSE 2010
  Section: 01
  Description of this file: Description of this file: Simulates a chat support system that will 
  match a customer's request to the available representative. Available representatives, customers 
  on hold, and active chat sessions are tracked with a separate SinglyLinkedList, as required by the 
  assignment. It will read a sequenece of events (timestamped) from the input file named on the 
  command line. Will then print the results from the event log. The output will follow the formats 
  given to us in the assignment description.

 */

/* 
      The file reader java tool is necessary to read the hw1in and hw1out txt files, but because it only
      reads one character at a time, the buffered reader tool is used to read bigger chunks at a time.
      The IOException java tool is used to ensure that if there is an issue reading the txt file or if it
      doesn't exist, the program will crash and provide an error.
*/

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class HW1
{
    /*
      HoldCustomer class is basically used to make it possible to put two values on
      a single node. This is because in a singly linked list, nodes can only hold one value. This
      class is basically being used as a container to hold multiple values and then be stored on a
      node as one value, containing multiple. This class will hold two pieces of information, the
      customers name and the time that they originally asked for a chat.
    */
    private static class HoldCustomer {
        String name;
        String requestTime;
        HoldCustomer(String name, String requestTime) {
            this.name = name;
            this.requestTime = requestTime;
        }
        public boolean equals(Object o) {
            if (!(o instanceof HoldCustomer)) return false;
            return name.equals(((HoldCustomer) o).name);
        }
    }

    /*
      Same as HoldCustomer, but insdtead of holding the customer name and time they asked for a chat
      , this class is going to hold which customer is chatting and which rep they are chatting with.
    */
    private static class ChatSession {
        String customer;
        String rep;
        ChatSession(String customer, String rep) {
            this.customer = customer;
            this.rep = rep;
        }
        public boolean equals(Object o) {
            if (!(o instanceof ChatSession)) return false;
            return customer.equals(((ChatSession) o).customer);
        }
    }

    /*
      This class is to turn the time wait times into minutes because if you were to try and subtract
      the time values in the 4 digit HHMM format that they are given in, you would end up with
      incorrect values. e.g. If you had 0930 and 1030, if you were to subract those, you would get 
      100, which isn't correct. Without this class, the outputs would be incorrect.
    */
    private static int toMinutes(String hhmm) {
        int hours = Integer.parseInt(hhmm.substring(0, 2));
        int minutes = Integer.parseInt(hhmm.substring(2, 4));
        return hours * 60 + minutes;
    }

    /*
      This class is necessary to take the minutes format back into the 4 digit HHMM format for the
      output. HHMM stands for Hours/Minutes.
    */
    private static String toHHMM(int totalMinutes) {
        int hours = totalMinutes / 60;
        int minutes = totalMinutes % 60;
        return String.format("%02d%02d", hours, minutes);
    }

    public static void main(String[] args) throws IOException
    {
	/* description of variables:
	   available  - a linked list of representative names (Strings) that are currently free to
                        take a chat.
	   onHold     - a linked list of HoldCustomer objects, one per customer that is currently
                        waiting on hold.
	   sessions   - a linked list of ChatSession objects, only one per chat that is currently in
                        progress
	   maxWaitMinutes - is a single running number to track the biggest wait time 
        */

	SinglyLinkedList<String> available = new SinglyLinkedList<>();
	SinglyLinkedList<HoldCustomer> onHold = new SinglyLinkedList<>();
	SinglyLinkedList<ChatSession> sessions = new SinglyLinkedList<>();
	int maxWaitMinutes = 0;

	// A Java array that is just used to list the representatives in order
	String[] initialReps = {"Alice", "Bob", "Carol", "David", "Emily"};
	for (String rep : initialReps)
	    available.addLast(rep);

	BufferedReader br = new BufferedReader(new FileReader(args[0]));
	StringBuilder out = new StringBuilder();
	String line;

	while ((line = br.readLine()) != null) {
	    line = line.trim();
	    if (line.isEmpty()) continue;
	    String[] tok = line.split("\\s+");
	    String type = tok[0];

	    /* 
               When a customer requests a chat, assign the front rep if available. Otherwise, put
               the customer on hold if they choose to wait or let them try later.               
            */
	    if (type.equals("ChatRequest")) {
		String requestTime = tok[1];
		String customer = tok[2];
		String waitOrLater = tok[3];
		out.append("ChatRequest ").append(requestTime).append(' ')
		   .append(customer).append(' ').append(waitOrLater).append('\n');

		if (!available.isEmpty()) {
		    String rep = available.removeFirst();
		    sessions.addLast(new ChatSession(customer, rep));
		    out.append("RepAssignment ").append(customer).append(' ')
		       .append(rep).append(' ').append(requestTime).append('\n');
		} else if (waitOrLater.equals("wait")) {
		    onHold.addLast(new HoldCustomer(customer, requestTime));
		    out.append("PutOnHold ").append(customer).append(' ')
		       .append(requestTime).append('\n');
		} else {
		    out.append("TryLater ").append(customer).append(' ')
		       .append(requestTime).append('\n');
		}

	    /*
               If a customer on hold gives up waiting, remove them from the hold list and then
               record their wait time.
            */
	    } else if (type.equals("QuitOnHold")) {
		String quitTime = tok[1];
		String customer = tok[2];
		out.append("QuitOnHold ").append(quitTime).append(' ')
		   .append(customer).append('\n');

		HoldCustomer removed = onHold.remove(new HoldCustomer(customer, null));
		if (removed != null) {
		    int wait = toMinutes(quitTime) - toMinutes(removed.requestTime);
		    if (wait > maxWaitMinutes) maxWaitMinutes = wait;
		}

	    /* 
               When a chat fininshes and the representative becomes free, if someone is on hold,
               immediatly the next, longest waiting, customer to the newly free representative and
               record the newly determined wait time. If there is no waiting customer, return the
               representative back to the available list.
            */
	    } else if (type.equals("ChatEnded")) {
		String customer = tok[1];
		String rep = tok[2];
		String endTime = tok[3];
		out.append("ChatEnded ").append(customer).append(' ')
		   .append(rep).append(' ').append(endTime).append('\n');
		sessions.remove(new ChatSession(customer, null));

		if (!onHold.isEmpty()) {
		    HoldCustomer next = onHold.removeFirst();
		    sessions.addLast(new ChatSession(next.name, rep));
		    out.append("RepAssignment ").append(next.name).append(' ')
		       .append(rep).append(' ').append(endTime).append('\n');
		    int wait = toMinutes(endTime) - toMinutes(next.requestTime);
		    if (wait > maxWaitMinutes) maxWaitMinutes = wait;
		} else {
		    available.addLast(rep);
		}

	    // Print the currently available representatives front to back.
	    } else if (type.equals("PrintAvailableRepList")) {
		String printTime = tok[1];
		StringBuilder sb = new StringBuilder("AvailableRepList ").append(printTime);
		for (String rep : available)
		    sb.append(' ').append(rep);
		out.append(sb).append('\n');

	    /* Print the largest determined wait time seen so far. */
	    } else if (type.equals("PrintMaxWaitTime")) {
		String printTime = tok[1];
		out.append("MaxWaitTime ").append(printTime).append(' ')
		   .append(toHHMM(maxWaitMinutes)).append('\n');
	    }
	}
	br.close();

	System.out.print(out);
    }

}
