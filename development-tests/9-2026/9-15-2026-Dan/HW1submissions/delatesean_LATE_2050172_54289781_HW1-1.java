/*

  Author: Sean Delate
  Email: sdelate2025@my.fit.edu
  Course: CSE2010
  Section: 01
  Description of this file: Recieves an input file and transforms it such that it becomes the output file

 */
import java.io.File; // Needed to read files
import java.io.FileNotFoundException; // Needed to handle errors
import java.util.Scanner; // Needed for text input
public class HW1
{
	public class Time { // This class is needed to do math on HH-MM time formats

		private int hours; // Tracks amount of hours

		private int minutes; // Tracks amount of minutes

		public Time(String x) {
			if (x == null || x.length() < 4) {
				hours = 0;
				minutes = 0;
				return;
			}
			hours = Integer.parseInt(x.substring(0, 2));
			minutes = Integer.parseInt(x.substring(2, 4));
		}


		// makes the time in minutes ad hours into a string
		public String format() {
			String sHours = "";
			String sMinutes = "" + minutes;
			// if the minutes or hours are under 10 it adds a 0 before to keep it in HH-MM
			if (hours < 10) { 
				sHours = "0" + hours;
			}
			if (minutes < 10) {
				sMinutes = "0" + minutes;
			}
			return (sHours + sMinutes);
		}
	}

	public class Customer { // This class just holds the name of the customer and when they requested to chat

		private String name; // Customer's name

		private Time requestTime; // Time request was sent

		public Customer() { // defult customer constructor
			name = "Defult";
			Time newTime = new Time("0000");
			requestTime = newTime;
		}

		// Sets a customers name and time they requested to chat
		public Customer(String name, String time) {
			this.name = name;
			requestTime = new Time(time);
		}

		// Getter methods for Customer
		public String name() {
			return name;
		}

		public Time requestTime() {
			return requestTime;
		}
	}

	public class Representative { // Creates a new representative class to contain who they are, if they are avalable, and the customer they are chatting to if they are not

		private String name; // Shows representative name - is originally set to Defult

		private Customer customer; // Holds the customer's name and time their request was sent
		
		public Representative(String name) { // Constructor sets name and makes avalable
			this.name = name;
			customer = new Customer();
		}

		//Getter methods for representative
		public String name() { 
			return name;
		}

		public Customer customer() {
			return customer;
		}


		public void chattingTo(String name, String time) { // Setter method for the customer name
			customer = new Customer(name, time);
		}

		public void makeAvalable() { // Sets customer to defult
			customer = new Customer();
		}
	}

	public Time maxWait = new Time("0000"); // Used to calculate max wait time

	public SinglyLinkedList<Representative> avalableReps = new SinglyLinkedList<>(); // Adds an initial empty SinglyLinkedList called avalableReps to contain representitives for easy access

	public SinglyLinkedList<Representative> chattingReps = new SinglyLinkedList<>(); // Contains all currently chatting representatives
	
	public SinglyLinkedList<Customer> waitingCustomers = new SinglyLinkedList<>(); // Contains all waiting customers

	// Takes a string input with spaces for the representatives
	// This could just be done with an array of names but a string input is easier to use as a person inputting names into the system
    public void repAssignment(String repList) { 
		// Splits the names separated by spaces into a string array
		String[] reps =  repList.split("\\s"); 
		// Places each name into the name field of the representative class then adds the new represenative object to the list of representatives
    	for (String rep : reps) {
			Representative newRep = new Representative(rep);
			avalableReps.addLast(newRep);
		}
	}

	// Adds waiting customers to the queue at the back of the line 
	// May be unneccesary but I thought the code looks more readable with it
	public void addToQueue(String n, String time) { 
		Customer newCustomer = new Customer(n, time);
		waitingCustomers.addLast(newCustomer);
		isMaxWait(newCustomer, time);
	}

		// Is a boolean method that finds if one time is bigger than another
	public boolean isMore(Time x, Time y) {
		return (x.hours*60+x.minutes) > (y.hours*60+y.minutes);
	}

	// subtracts one time from another
	public Time subtractTime(Time x, Time y) {
		int newMinutes = ((x.hours*60+x.minutes) - (y.hours*60+y.minutes));
		Time newTime = new Time("");
		newTime.hours = newMinutes / 60;
		newTime.minutes = newMinutes % 60;
		return newTime;
	}

	// Detects if the request time for the customer is longer than the current max wait
	public void isMaxWait(Customer c, String time) {
		if (c == null || c.requestTime() == null) {
			return;
		}
		Time currentTime = new Time(time);
		Time waitTime = subtractTime(currentTime, c.requestTime());
		if (isMore(waitTime, maxWait)) {
			maxWait = waitTime;
		}
	}

	// Moves reps from avalableReps to chattingReps when they are avalable and adds a customer to their customer field
	public void repRequested(String n, String time, boolean wait) { 
		// If there are reps avalable it will just assign the customer to one
		if (!(avalableReps.isEmpty())) {
			chattingReps.addFirst(avalableReps.first());
			chattingReps.first().chattingTo(n, time);
			avalableReps.removeFirst();
			System.out.println("RepAssignment " + n + " " + chattingReps.first().name() + " " + time);
		} else if (wait) { 	// Otherwise if they can wait they will be put into the waiting customer queue
			addToQueue(n, time);
			System.out.println("PutOnHold " + " " + n + " " + time);
		} else {  // Otherwise if they cant wait they will be told to try again later
			System.out.println("TryLater " + " " + n + " " + time);
		}
	}

	// Checks if there are avalable representatives and if there are assigns one of the customers in queue to them
	public void checkAndMove(String time) { 
		if (!(avalableReps.isEmpty()) && !(waitingCustomers.isEmpty())) {
			Customer next = waitingCustomers.removeFirst();
			isMaxWait(next, time);
			repRequested(next.name(), time, false);
		}
	}

	// Removes representatives from finished conversations in the chattingReps list, makes them avalable, and adds them back to avalableReps
	public void chatEnded(String n, String time) { 
		// The for loop iterates through the linked list by removing the last object and adding it to the front. 
		// This was the best way I could think to do this without changing the SinglyLinkedList class.
		for (int i = 0; i < chattingReps.size(); i++) { 
			if (chattingReps.first().customer().name().equals(n)) {
				chattingReps.first().makeAvalable();
				avalableReps.addLast(chattingReps.first());
				chattingReps.removeFirst();
				break;
			}
			Representative temp = chattingReps.removeFirst();
			chattingReps.addLast(temp);
		}
		if (!(avalableReps.isEmpty()) && !(waitingCustomers.isEmpty())) {
			checkAndMove(time);
		}
	}

	// Finds and removes a customer from the queue
	public void quitOnHold(String n, String time) { 
		// Uses the same for loop searching as chatEnded
		for (int i = 0; i < waitingCustomers.size(); i++) {
			if (waitingCustomers.first().name().equals(n)) {
				isMaxWait(waitingCustomers.first(), time);
				waitingCustomers.removeFirst();
				break;
			}
			Customer temp = waitingCustomers.removeFirst();
			waitingCustomers.addLast(temp);
		}
		// Prints the command used, time, and name of customer
		System.out.println("QuitOnHold " + time + " " + n);
	}

	// Prints a list of avalable represenatatives
	public void printAvalableList(String time) { 
		System.out.print("AvailableRepList " + time);
		// Uses the same loop search as chatEnded
		for (int i = 0; i < avalableReps.size(); i++) {
			Representative temp = avalableReps.removeFirst();
			System.out.print(" " + temp.name());
			avalableReps.addLast(temp);
		}
		System.out.println();
	}

	// Prints a string array for some of the commands
	public void printString(String[] sArr) {
		for (String arr : sArr) {
			System.out.print(arr + " ");
		}
		System.out.println();
	}

	// Prints the maximum time that a customer will have to wait
	public void printMaxWaitTime(String time) {
		System.out.println("MaxWaitTime " + time + " " +  maxWait.format());
	}

    public static void main(String[] args)
    {

		HW1 systemRequests = new HW1(); // Used to simulate chat requests

		File instructions = new File(args[0]); // Takes the command line arguement for the file and then uses it to make a new file object to be read later

		// assigns initial avalable representatives
		systemRequests.repAssignment("Alice Bob Carol David Emily");

		// A try catch is needed to handle if the file with the chat commands cannot be found
		try (Scanner scan = new Scanner(instructions)) {
			// Looks through the file line by line
			while (scan.hasNextLine()) {
				// Reads the full line of text as a string input and then splits it into an array by spaces
				String data = scan.nextLine();
				String[] chatCommand = data.split("\\s");
					// Looks at the first element of the string array and depending on the command excecutes a method
					switch (chatCommand[0]) {
						case "ChatRequest":
							systemRequests.printString(chatCommand);
							systemRequests.repRequested(chatCommand[2], chatCommand[1], chatCommand[3].equals("wait"));
							break;
						case "QuitOnHold":
							systemRequests.printString(chatCommand);
							systemRequests.quitOnHold(chatCommand[2], chatCommand[1]);
							break;
						case "ChatEnded":
							systemRequests.printString(chatCommand);
							systemRequests.chatEnded(chatCommand[1], chatCommand[3]);
							break;
						case "PrintAvailableRepList":
							systemRequests.printAvalableList(chatCommand[1]);
							break;
						case "PrintMaxWaitTime":
							systemRequests.printMaxWaitTime(chatCommand[1]);
							break;
						default:
							break;
					}
			}
		} catch (FileNotFoundException e) {
			System.out.println("Error: file not found");
			e.printStackTrace();
		}
    }

}
