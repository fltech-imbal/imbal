/*
Author: Airen Sharber
Email: asharber2025@my.fit.edu
Course: CSE 2010
Section: 1
Description: Takes input in the form of customers, representatives, 
             and wait time and pairs customers with available representatives 
             while keeping track of how long the customer has been
             waiting using three linked lists with three different nodes.
             It also uses helper methods processChatRequest, processChatEnded, 
             processQuitOnHold, printAvailableRepList, printMaxWaitTime,
             updateMaxWaitTime, timeToMinutes, and formatTime. These methods help process
             events and sort where customers or reps should be placed, printing
             available reps, updating and printing the max wait time, and converting
             time into minutes to allow time to be calculated.
*/
import java.util.Scanner;
import java.io.*;
public class HW1 {
  public static void main(String[] args) throws Exception {
    Scanner fileReader = new Scanner(new File(args[0]));

    RepList availableReps = new RepList();
    CustomerList customersOnHold = new CustomerList();
    ChatList chatSessions = new ChatList();

    int maxWaitTimeSoFar = 0;

    availableReps.add("Alice");
    availableReps.add("Bob");
    availableReps.add("Carol");
    availableReps.add("David");
    availableReps.add("Emily");

    while (fileReader.hasNext()) {
      String eventType = fileReader.next();
      if (eventType.equals("ChatRequest")) {
        int time = fileReader.nextInt();
        String customer = fileReader.next();
        String waitOrLater = fileReader.next();
        maxWaitTimeSoFar = processChatRequest(time, customer, waitOrLater,
                                              availableReps, customersOnHold,
                                              chatSessions, maxWaitTimeSoFar);
      } else if (eventType.equals("ChatEnded")) {
        String customer = fileReader.next();
        String rep = fileReader.next();
        int time = fileReader.nextInt();
        maxWaitTimeSoFar = processChatEnded(time, customer, rep, availableReps,
                                            customersOnHold, chatSessions, maxWaitTimeSoFar);
      } else if (eventType.equals("QuitOnHold")) {
        int time = fileReader.nextInt();
        String customer = fileReader.next();
        maxWaitTimeSoFar = processQuitOnHold(time, customer, customersOnHold, maxWaitTimeSoFar);
      } else if (eventType.equals("PrintAvailableRepList")) {
        int time = fileReader.nextInt();
        printAvailableRepList(time, availableReps);
      } else if (eventType.equals("PrintMaxWaitTime")) {
        int time = fileReader.nextInt();
        printMaxWaitTime(time, maxWaitTimeSoFar);
      }
    }

    fileReader.close();    
  }
  static int processChatRequest(int time, String customer,String waitOrLater,
                                 RepList availableReps, CustomerList customersOnHold,
                                 ChatList chatSessions, int maxWaitTimeSoFar) {
    System.out.println("ChatRequest " + formatTime(time) + " " + customer + " " + waitOrLater);
    if (!availableReps.isEmpty()) {
      String rep = availableReps.remove();
      chatSessions.add(customer, rep, time);
      System.out.println("RepAssignment " + customer + " " + rep + " " + formatTime(time));
    } else if (waitOrLater.equals("wait")) {
      customersOnHold.add(customer, time);
      System.out.println("PutOnHold " + customer + " " + formatTime(time));
    } else {
      System.out.println("TryLater " + customer + " " + formatTime(time));
    }
    return maxWaitTimeSoFar;
  }
  static int processChatEnded(int time, String customer, String rep, RepList availableReps,
                              CustomerList customersOnHold, ChatList chatSessions, int
                              maxWaitTimeSoFar) {
    System.out.println("ChatEnded " + customer + " " + rep + " " + formatTime(time));
    chatSessions.remove(customer);
    if (!customersOnHold.isEmpty()) {
      CustomerNode customerNode = customersOnHold.removeFirst();
      chatSessions.add(customerNode.customer, rep, time);
      System.out.println("RepAssignment " + customerNode.customer + " " + rep + " " + 
                         formatTime(time));
      maxWaitTimeSoFar = updateMaxWaitTime(customerNode.requestTime, time, maxWaitTimeSoFar);
    } else {
      availableReps.add(rep);
    }
    return maxWaitTimeSoFar;
  }
  static int processQuitOnHold(int time, String customer, CustomerList customersOnHold,
                               int maxWaitTimeSoFar) {
    CustomerNode customerNode = customersOnHold.remove(customer);
    System.out.println("QuitOnHold " + formatTime(time) + " " + customer);
    maxWaitTimeSoFar = updateMaxWaitTime(customerNode.requestTime, time, maxWaitTimeSoFar);
    return maxWaitTimeSoFar;
  }
  static void printAvailableRepList(int time, RepList availableReps) {
    System.out.print("AvailableRepList " + formatTime(time));
    availableReps.print();
    System.out.println();
  }
  static void printMaxWaitTime(int time, int maxWaitTimeSoFar) {
    int hours = maxWaitTimeSoFar / 60;
    int minutes = maxWaitTimeSoFar % 60;
    System.out.printf("MaxWaitTime %04d %02d%02d%n", time, hours, minutes);
  }
  static int updateMaxWaitTime(int requestTime, int endTime, int maxWaitTimeSoFar) {
    int waitTime = timeToMinutes(endTime) - timeToMinutes(requestTime);
    if (waitTime > maxWaitTimeSoFar) {
      return waitTime;
    }
    return maxWaitTimeSoFar;
  }
  static int timeToMinutes(int time) {
    int hours = time / 100;
    int minutes = time % 100;
    return hours * 60 + minutes;
  }
  static String formatTime(int time) {
    return String.format("%04d", time);
  }
}
class RepList {
  private RepNode head;
  private RepNode tail;
  public RepList() {
    head = null;
    tail = null;
  }
  public boolean isEmpty() {
    return head == null;
  }
  public void add(String rep) {
    RepNode newNode = new RepNode(rep);
    if (isEmpty()) {
      head = newNode;
      tail = newNode;
    } else {
      tail.next = newNode;
      tail = newNode;
    }
  }
  public String remove() {
    if (isEmpty()) {
      return null;
    }
    String rep = head.rep;
    head = head.next;
    if (head == null) {
      tail = null;
    }
    return rep;
  }
  public String peek() {
    if (isEmpty()) {
      return null;
    }
    return head.rep;
  }
  public void print() {
    RepNode current = head;
    while (current != null) {
      System.out.print(" " + current.rep);
      current = current.next;
    }
  }
}
class RepNode {
  String rep;
  RepNode next;
  public RepNode (String rep) {
    this.rep = rep;
    this.next = null;
  }
}
class CustomerList {
  private CustomerNode head;
  private CustomerNode tail;
  public CustomerList() {
    head = null;
    tail = null;
  }
  public boolean isEmpty() {
    return head == null;
  }
  public void add(String customer, int requestTime) {
    CustomerNode newNode = new CustomerNode(customer, requestTime);
    if (isEmpty()) {
      head = newNode;
      tail = newNode;
    } else { 
      tail.next = newNode;
      tail = newNode;
    }
  }
  public CustomerNode removeFirst() {
    if (isEmpty()) {
      return null;
    }
    CustomerNode removed = head;
    head = head.next;
    if (head == null) {
      tail = null;
    }
    removed.next = null;
    return removed;
  }
  public CustomerNode remove(String customer) {
    if (isEmpty()) {
      return null;
    }
    if (head.customer.equals(customer)) {
      return removeFirst();
    }
    CustomerNode current = head;
    while (current.next != null) {
      if (current.next.customer.equals(customer)) {
        CustomerNode removed = current.next;
        current.next = removed.next;
        if (removed == tail) {
         tail = current;
        }
        removed.next = null;
        return removed;
      }
      current = current.next;
    }
    return null;
  }
  public CustomerNode peek() {
    if (isEmpty()) {
      return null;
    }
    return head;
  }
  public void print() {
    CustomerNode current = head;
    while (current != null) {
      System.out.print(" " + current.customer);
      current = current.next;
    }
  }
}
class CustomerNode {
  String customer;
  int requestTime;
  CustomerNode next;
  public CustomerNode (String customer, int requestTime) {
    this.customer = customer;
    this.requestTime = requestTime;
    this.next = null;
  }
}
class ChatList {
  private ChatNode head;
  public ChatList() {
    head = null;
  }
  public boolean isEmpty() {
    return head == null;
  }
  public void add(String customer, String rep, int requestTime) {
    ChatNode newNode = new ChatNode(customer, rep, requestTime);
    newNode.next = head;
    head = newNode;
  }
  public ChatNode find(String customer) {
    ChatNode current = head;
    while (current != null) {
      if (current.customer.equals(customer)) {
        return current;
      }
      current = current.next;
    }
    return null;
  }
  public ChatNode remove(String customer) {
    if (head == null) {
      return null;
    }
    if (head.customer.equals(customer)) {
      ChatNode removed = head;
      head = head.next;
      removed.next = null;
      return removed;
    }
    ChatNode current = head;
    while (current.next != null) {
      if (current.next.customer.equals(customer)) {
        ChatNode removed = current.next;
        current.next = removed.next;
        removed.next = null;
        return removed;
      }
      current = current.next;
    }
    return null;
  }
  public void print() {
    ChatNode current = head;
    while (current != null) {
      System.out.print(" " + current.customer);
      current = current.next;
    }
  }
}
class ChatNode {
  String customer;
  String rep;
  int requestTime;
  ChatNode next;
  public ChatNode (String customer, String rep, int requestTime) {
    this.customer = customer;
    this.rep = rep;
    this.requestTime = requestTime;
    this.next = null;
  }
}
