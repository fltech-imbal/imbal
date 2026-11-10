/*

  Author: Sean Delate
  Email: sdelate2025@my.fit.edu
  Course: CSE2010
  Section: 01
  Description of this file: Create a timeline of major events in a users life.
  Using many events the user can specify a time range to display the major events.

  */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList; // Needed to read files
import java.util.Scanner; // Needed to handle errors
public class HW5<K extends Comparable<K>>
{
    //Class for the inserted nodes
    public class Node<K, V> {
        private K key; //The thing used for identification or the date
        private V value; // The event

        // used to set up the list
        // I used these as it seemed easier than doing it with the nested linkedlist
        private Node<K, V> under; 
        private Node<K, V> next;
        private Node<K, V> before;
        private Node<K, V> above;

        //Defult constructor
        public Node() {
            key = null;
            value = null;
        }

        //Initializes node
        public Node(K k, V v) {
            key = k;
            value = v;
        }

        //getters and setters
        public K key() { 
            return key;
        }

        public V value() {
            return value;
        }

        public void setValue(V v) {
            value = v;
        }
    }
    
    public class SkipList<V> {
        private int maxHeight; //The max height the skiplist can reach
        private int height; //The current height of the skiplist
        private FakeRandHeight ranHeight; //the false random list used
        private Node<Integer, V> head; //the top left of the list

        //defult constructor
        public SkipList() {
            maxHeight = 10;
            height = 0;
            ranHeight = new FakeRandHeight();

            Node<Integer, V> minNode = new Node<>(Integer.MIN_VALUE, null);
            Node<Integer, V> maxNode = new Node<>(Integer.MAX_VALUE, null);
            minNode.next = maxNode;
            maxNode.before = minNode;
            head = minNode;
        }

        //gets the node with value k
        public Node<Integer, V> get(int k) {
            Node<Integer, V> current = head;
            while (current != null) {
                while (current.next != null && current.next.key < k) {
                    current = current.next;
                }
                if (current.next != null && current.next.key == k) {
                    return current.next;
                }
                current = current.under;
            }
            return null;
        }

        //places a new node with value v and key k
        public V put(int k, V value) {
            Node<Integer, V> found = get(k);
            if (found != null) {
                V oldValue = found.value();
                while (found.under != null) {
                    found = found.under;
                }
                while (found != null) {
                    found.setValue(value);
                    found = found.above;
                }
                return oldValue;
            }

            //increases height of the list if the value given is higher than the current height
            int nodeHeight = Math.min(ranHeight.get(), maxHeight - 1);
            while (height < nodeHeight) {
                Node<Integer, V> oldTail = head;
                while (oldTail.next != null) {
                    oldTail = oldTail.next;
                }

                Node<Integer, V> newHead = new Node<>(Integer.MIN_VALUE, null);
                Node<Integer, V> newTail = new Node<>(Integer.MAX_VALUE, null);
                newHead.next = newTail;
                newTail.before = newHead;
                newHead.under = head;
                newTail.under = oldTail;

                head = newHead;
                height++;
            }

            Node<Integer, V> lowerNode = null;
            for (int i = 0; i <= nodeHeight; i++) {
                Node<Integer, V> largestBefore = largestBefore(k, i);
                Node<Integer, V> newNode = new Node<>(k, value);
                newNode.before = largestBefore;
                newNode.next = largestBefore.next;
                largestBefore.next.before = newNode;
                largestBefore.next = newNode;
                newNode.under = lowerNode;
                if (lowerNode != null) {
                    lowerNode.above = newNode;
                }
                lowerNode = newNode;
            }
            // if the list doesnt have a top of min and max increases height and adds another layer
            if (head.next.key != Integer.MAX_VALUE) {
                Node<Integer, V> oldTail = head;
                while (oldTail.next != null) {
                    oldTail = oldTail.next;
                }

                Node<Integer, V> newHead = new Node<>(Integer.MIN_VALUE, null);
                Node<Integer, V> newTail = new Node<>(Integer.MAX_VALUE, null);
                newHead.next = newTail;
                newTail.before = newHead;
                newHead.under = head;
                newTail.under = oldTail;

                head = newHead;
                height++;
            }
            return null;
        }

        //Finds the largest node before the curent one for use in the put in order to find where the copy needs to be
        //when adding an object to multiple levels
        private Node<Integer, V> largestBefore(int k, int level) {
            Node<Integer, V> current = head;
            for (int i = height; i > level; i--) {
                while (current.next.key < k) {
                    current = current.next;
                }
                current = current.under;
            }
            while (current.next.key < k) {
                current = current.next;
            }
            return current;
        }

        // removes a node and if it is the only node in the layer removes the layer
        public V remove(int k) {
            Node<Integer, V> found = get(k);
            if (found == null) {
                return null;
            }

            V foundValue = found.value();

            while (found.under != null) {
                found = found.under;
            }

            while (found != null) {
                Node<Integer, V> above = found.above;
                found.before.next = found.next;
                found.next.before = found.before;
                found = above;
            }

            while (head.under != null
                    && head.next.key == Integer.MAX_VALUE
                    && head.under.next.key == Integer.MAX_VALUE) {
                head = head.under;
                height--;
            }

            return foundValue;
        }

        // Finds the entry with the largest key before the current one 
        public Node<Integer, V> floorEntry(int k) {
            Node<Integer, V> found = get(k);
            if (found == null) {
                return null;
            }
            while (found.under != null) {
                found = found.under;
            }
            if (found.before == null) {
                return null;
            }
            return found.before;
        }

        //Finds the entry with the smallest key after the current one
        public Node<Integer, V> ceilingEntry(int k) {
            Node<Integer, V> found = get(k);
            if (found == null) {
                return null;
            }
            while (found.under != null) {
                found = found.under;
            }
            if (found.before == null) {
                return null;
            }
            return found.next;
        }

        //Creates a list of the objects between the two keys 
        // is currently unused because i found it easier to use a different method
        public ArrayList<Node<Integer, V>> subMap(int k1, int k2) {
            Node<Integer, V> leftNode = get(k1);
            Node<Integer, V> rightNode = get(k2);
            if (leftNode == null || rightNode == null) {
                return null;
            }
            ArrayList<Node<Integer, V>> map = new ArrayList<>();
            while (leftNode.under != null) {
                leftNode = leftNode.under;
            }
            while (rightNode.under != null) {
                rightNode = rightNode.under;
            }
            while (leftNode != rightNode) {
                map.add(leftNode);
                leftNode = leftNode.next;
            }
            map.add(leftNode);
            return map;
        }
    }

    private SkipList<String> head = new SkipList<>(); //the head list

    //Converts dates into days
    public int dateConvert(int date) {
        int month = date / 100;
        int day = date % 100;
        return (month - 1) * 31 + day;
    }

    //Converts days into dates
    public String daysConvert(int days) {
        int month = (days - 1) / 31 + 1;
        int day = (days - 1) % 31 + 1;
        return String.format("%02d%02d", month, day);
    }

    //Displays an event
    //most of the following methods are just checks to see if an entry is null and then a print
    public void displayEvent(int date) {
        if (head.get(dateConvert(date)) != null) {
            System.out.print(" " + head.get(dateConvert(date)).value());
        } else {
            System.out.print(" none");
        }
    }

    //adds an event
    public void addEvent(int date, String event) {
        head.put(dateConvert(date), event);
        System.out.print(" success");
    }

    //deletes an event
    public void deleteEvent(int date) {
        if (head.remove(dateConvert(date)) != null) {
            System.out.print(" success");
        } else {
            System.out.print(" noDateError");
        }
    }

    //Prints all events and dates between two points by moving down the first columndescribed by the 
    // start date and then printing all objects until the end date
    public void displayEventsBetweenDates(int startDate, int endDate) {
        int startKey = dateConvert(startDate);
        int endKey = dateConvert(endDate);

        Node<Integer, String> current = head.head;
        while (current.under != null) {
            current = current.under;
        }
        current = current.next;

        boolean printed = false;
        while (current != null && current.key != Integer.MAX_VALUE) {
            if (current.key >= startKey && current.key <= endKey) {
                System.out.print(" " + daysConvert(current.key) + ":" + current.value);
                printed = true;
            }
            current = current.next;
        }

        if (!printed) {
            System.out.print(" none");
        }
    }

    //Same thing as the previous one but goes from the start date to the max
    public void displayEventsFromStartDate(int startDate) {
        int startKey = dateConvert(startDate);

        Node<Integer, String> current = head.head;
        while (current.under != null) {
            current = current.under;
        }
        current = current.next;

        boolean printed = false;
        while (current != null && current.key != Integer.MAX_VALUE) {
            if (current.key >= startKey) {
                System.out.print(" " + daysConvert(current.key) + ":" + current.value);
                printed = true;
            }
            current = current.next;
        }

        if (!printed) {
            System.out.print(" none");
        }
    }

    //Same thing as the display between but goes from head to end date
    public void displayEventsToEndDate(int endDate) {
        int startKey = dateConvert(endDate);

        Node<Integer, String> current = head.head;
        while (current.under != null) {
            current = current.under;
        }
        current = current.next;

        boolean printed = false;
        while (current != null && current.key != startKey) {
            if (current.key <= startKey) {
                System.out.print(" " + daysConvert(current.key) + ":" + current.value);
                printed = true;
            }
            current = current.next;
        }

        if (!printed) {
            System.out.print(" none");
        }
    }

    //Same as ones previous but goes from min to max and prints on the way but doesnt print min or max
    public void displayAllEvents() {
        Node<Integer, String> current = head.head;
        while (current.under != null) {
            current = current.under;
        }
        current = current.next;

        boolean printed = false;
        while (current != null && current.key != Integer.MAX_VALUE) {
            System.out.print(" " + daysConvert(current.key) + ":" + current.value);
            printed = true;
            current = current.next;
        }

        if (!printed) {
            System.out.print(" none");
        }
    }

    //Prints everything in the list by going from head to min and then down and then repeating
    public void printSkipList() {
        Node<Integer, String> rowHead = head.head;
        int level = head.height;
    
        while (rowHead != null) {
            System.out.print("(S" + level + ")");
            Node<Integer, String> current = rowHead.next;
            boolean printed = false;
    
            while (current != null && current.key != Integer.MAX_VALUE) {
                System.out.print(" " + daysConvert(current.key) + ":" + current.value);
                printed = true;
                current = current.next;
            }
    
            if (!printed) {
                System.out.print(" empty");
            }
            System.out.println();
    
            rowHead = rowHead.under;
            level--;
        }
    }

    public static void main(String[] args)
    {
        HW5<String> systemRequests = new HW5<>(); // Used to run methods

		File data = new File(args[0]); // Takes the command line arguement for the file and then uses it to make a new file object to be read later

        // A try catch is needed to handle if the file with the chat commands cannot be found
		try (Scanner scan = new Scanner(data)) {
			// Looks through the file line by line
			while (scan.hasNextLine()) {
				// Reads the full line of text as a string input and then splits it into an array by spaces
				String info = scan.nextLine();
				String[] instruction = info.split("\\s");
                int date;
					// Looks at the first element of the string array and depending on the command excecutes a method
                    // usually prints names and instructions before the instruction results
					switch (instruction[0]) {
						case "DisplayEvent":
                            System.out.print(info);
                            date = Integer.parseInt(instruction[1]);
                            systemRequests.displayEvent(date);
                            System.out.println();
                            break;
                        case "AddEvent":
                            System.out.print(info);
                            date = Integer.parseInt(instruction[1]);
                            systemRequests.addEvent(date, instruction[2]);
                            System.out.println();
                            break;
                        case "DeleteEvent":
                            System.out.print(info);
                            date = Integer.parseInt(instruction[1]);
                            systemRequests.deleteEvent(date);
                            System.out.println();
                            break;
                        case "DisplayEventsBetweenDates":
                            System.out.print(info);
                            systemRequests.displayEventsBetweenDates(Integer.parseInt(instruction[1]), Integer.parseInt(instruction[2]));
                            System.out.println();
                            break;
                        case "DisplayEventsFromStartDate":
                            System.out.print(info);
                            date = Integer.parseInt(instruction[1]);
                            systemRequests.displayEventsFromStartDate(date);
                            System.out.println();
                            break;
                        case "DisplayEventsToEndDate":
                            System.out.print(info);
                            date = Integer.parseInt(instruction[1]);
                            systemRequests.displayEventsToEndDate(date);
                            System.out.println();
                            break;
                        case "DisplayAllEvents":
                            System.out.print(info);
                            systemRequests.displayAllEvents();
                            System.out.println();
                            break;
                        case "PrintSkipList":
                            System.out.println(info);
                            systemRequests.printSkipList();
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
