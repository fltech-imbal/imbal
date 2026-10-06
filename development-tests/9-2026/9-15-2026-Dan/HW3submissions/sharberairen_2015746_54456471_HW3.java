/*
Author: Airen Sharber
Email: asharber2025@code01.fit.edu
Course: CSE 2010
Section: 1
Description:Builds an organizational tree and answers queries about
            supervisors, subordinates, ranks, and relationships.

*/
import java.util.*;
import java.io.*;
public class HW3 {
  public static void main(String[] args) throws Exception {
    Scanner dataFile = new Scanner(new File(args[0]));
    String rootName = dataFile.nextLine();
    Tree tree = new Tree(rootName);
    while (dataFile.hasNextLine()) {
      String line = dataFile.nextLine();
      if (!line.isEmpty()) {
        String[] parts = line.split(" ");
        String supervisor = parts[0];
        String subordinate = parts[1];
        tree.addChild(supervisor, subordinate);
      }
    }
    dataFile.close();
    Scanner queryFile = new Scanner(new File(args[1]));
    while (queryFile.hasNextLine()) {
      String line = queryFile.nextLine();
      if (line.isEmpty()) {
        continue;
      }
      String[] parts = line.split(" ");
      String command = parts[0];
      if (command.equals("DirectSupervisor")) {
        directSupervisor(tree, parts[1]);
      } else if (command.equals("DirectSubordinates")) {
        directSubordinates(tree, parts[1]);
      } else if (command.equals("AllSupervisors")) {
        allSupervisors(tree, parts[1]);
      } else if (command.equals("AllSubordinates")) {
        allSubordinates(tree, parts[1]);
      } else if (command.equals("NumberOfAllSupervisors")) {
        numberOfAllSupervisors(tree, parts[1]);
      } else if (command.equals("NumberOfAllSubordinates")) {
        numberOfAllSubordinates(tree, parts[1]);
      } else if (command.equals("IsSupervisor")) {
        isSupervisor(tree, parts[1], parts[2]);
      } else if (command.equals("IsSubordinate")) {
        isSubordinate(tree, parts[1], parts[2]);
      } else if (command.equals("CompareRank")) {
        compareRank(tree, parts[1], parts[2]);
      } else if (command.equals("ClosestCommonSupervisor")) {
        closestCommonSupervisor(tree, parts[1], parts[2]);
      }
    }
    queryFile.close();
  }
  static void directSupervisor(Tree tree, String name) {
    Node node = tree.nodes.get(name);
    System.out.print("DirectSupervisor " + name + " ");
    if (node.parent == null) {
      System.out.println("none");
    } else {
      System.out.println(node.parent.name);
    }
  }
  static void directSubordinates(Tree tree, String name) {
    Node node = tree.nodes.get(name);
    System.out.print("DirectSubordinates " + name);
    if (node.children.isEmpty()) {
      System.out.println(" none");
      return;
    }
    for (Node child : node.children) {
      System.out.print(" " + child.name);
    }
    System.out.println();
  }
  static void allSupervisors(Tree tree, String name) {
    Node node = tree.nodes.get(name);
    System.out.print("AllSupervisors " + name);
    Node current = node.parent;
    if (current == null) {
      System.out.println(" none");
      return;
    }
    while (current != null) {
      System.out.print(" " + current.name);
      current = current.parent;
    }
    System.out.println();
  }
  static void allSubordinates(Tree tree, String name) {
    Node node = tree.nodes.get(name);
    System.out.print("AllSubordinates " + name);
    if (node.children.isEmpty()) {
      System.out.println(" none");
      return;
    }
    printAllSubordinates(node);
    System.out.println();
  }
  static void printAllSubordinates(Node node) {
    for (Node child : node.children) {
      System.out.print(" " + child.name);
      printAllSubordinates(child);
    }
  }
  static void numberOfAllSupervisors(Tree tree, String name) {
    Node node = tree.nodes.get(name);
    int count = 0;
    Node current = node.parent;
    while (current != null) {
      count++;
      current = current.parent;
    }
    System.out.println("NumberOfAllSupervisors " + name + " " + count);
  }
  static int countSubordinates(Node node) {
    int count = 0;
    for (Node child : node.children) {
      count++;
      count += countSubordinates(child);
    }
    return count;
  }
  static void numberOfAllSubordinates(Tree tree, String name) {
    Node node = tree.nodes.get(name);
    int count = countSubordinates(node);
    System.out.println("NumberOfAllSubordinates " + name + " " + count);
  }
  static void isSupervisor(Tree tree, String entity, String supervisor) {
    Node node = tree.nodes.get(entity);
    Node current = node.parent;
    while (current != null) {
      if (current.name.equals(supervisor)) {
        System.out.println("IsSupervisor " + entity + " " + supervisor + " yes");
        return;
      }
      current = current.parent;
    }
    System.out.println("IsSupervisor " + entity + " " + supervisor + " no");
  }
  static boolean findSubordinate(Node node, String name) {
    for (Node child : node.children) {
      if (child.name.equals(name)) {
        return true;
      }
      if (findSubordinate(child, name)) {
        return true;
      }
    }
    return false;
  }
  static void isSubordinate(Tree tree, String entity, String subordinate) {
    Node node = tree.nodes.get(entity);
    if (findSubordinate(node, subordinate)) {
      System.out.println("IsSubordinate " + entity + " " + subordinate + " yes");
    } else {
      System.out.println("IsSubordinate " + entity + " " + subordinate + " no");
    }
  }
  static int getDepth(Node node) {
    int depth = 0;
    while (node.parent != null) {
      depth++;
      node = node.parent;
    }
    return depth;
  }
  static void compareRank(Tree tree, String name1, String name2) {
    Node node1 = tree.nodes.get(name1);
    Node node2 = tree.nodes.get(name2);
    int depth1 = getDepth(node1);
    int depth2 = getDepth(node2);
    System.out.print("CompareRank " + name1 + " " + name2 + " ");
    if (depth1 < depth2) {
      System.out.println("higher");
    } else if (depth1 > depth2) {
      System.out.println("lower");
    } else {
      System.out.println("same");
    }
  }
  static void closestCommonSupervisor(Tree tree, String name1, String name2) {
    Node node1 = tree.nodes.get(name1);
    Node node2 = tree.nodes.get(name2);
    ArrayList<Node> supervisors1 = new ArrayList<>();
    Node current = node1.parent;
    while (current != null) {
      supervisors1.add(current);
      current = current.parent;
    }
    current = node2.parent;
    while (current != null) {
      if (supervisors1.contains(current)) {
        System.out.println("ClosestCommonSupervisor " + name1 + " " + name2 
                            + " " + current.name);
        return;
      }
      current = current.parent;
    }
    System.out.println("ClosestCommonSupervisor " + name1 + " " + name2 + " none");
  }
  static class Node {
    String name;
    Node parent;
    ArrayList<Node> children;
    Node(String name) {
      this.name = name;
      this.parent = null;
      this.children = new ArrayList<>();
    }
  }
  static class Tree {
    Node root;
    HashMap<String, Node> nodes;
    Tree(String rootName) {
      root = new Node(rootName);
      nodes = new HashMap<>();
      nodes.put(rootName, root);
    }
    void addChild(String parentName, String childName) {
      Node parent = nodes.get(parentName);
      if (parent == null) {
        parent = new Node(parentName);
        nodes.put(parentName, parent);
      }
      Node child = nodes.get(childName);
      if (child == null) {
        child = new Node(childName);
        nodes.put(childName, child);
      }
      child.parent = parent;
      parent.children.add(child);
      parent.children.sort((a,b) -> a.name.compareTo(b.name));
    }
    Node getParent(Node node) {
      return node.parent;
    }
    ArrayList<Node> getChildren(Node node) {
      return node.children;
    }
  }
}
