/*
 * Author: Jackson Young
 * Email: young2025@my.fit.edu
 * Course: CSE 2010
 * Section: 9:30am lab
 * Description of this file: This class functions like a Tree data structure using a linked list.
 * Each Tree will hold pointers to it's value and parent and children subtrees along with a depth value
 * Methods exist for adding children to a Tree, and getting the children or parent of a Tree
*/
import java.util.ArrayList;
import java.util.Comparator;
public class Tree<E> {
    
    // Pointers for maintaining the Tree
    Tree<E> parent; // The Supertree of this Tree
    ArrayList<Tree<E>> children; // The Subtrees of this Tree
    E value; // The value of this Tree node
    int depth; // The depth of this node down a super tree

    // Constructors
    public Tree(E value) { // Parent is null
        this.value = value;
        this.parent = null;
        this.children = new ArrayList();
        this.depth = 0;
    }
    public Tree(E value, Tree<E> parent) { // Parent is assigned
        this.value = value;
        this.parent = parent;
        this.children = new ArrayList();
        this.depth = parent.depth + 1;
    }

    // Public method for adding a child to the Tree
    public void addChild(E parent, E child) {
        Tree<E> parentNode = find(parent);
        parentNode.children.add(new Tree(child, parentNode));
    }

    // Public method for getting the children of a Tree
    public ArrayList<Tree<E>> getChildren(E subtree) {
        return find(subtree).children;
    }

    // Public method for getting the parent of a Tree
    public Tree<E> getParent(E subtree) {
        return find(subtree).parent;
    }

    // Public method used to recursively search the Tree for the specified Subtree
    // Returns null if the specified subtree is not found
    public Tree<E> find(E subtree) {
        // Check self (Base Case)
        if (this.value.equals(subtree)) {
            return this;
        }
        // Check children (Recursive Case)
        else {
            for (Tree<E> child : this.children) {
                Tree<E> foundValue = child.find(subtree);
                if (foundValue != null) return foundValue;
            }
        }
        return null;
    }

    // Public method used to get every child of a specified subtree
    // The child nodes are put into a list that is returned.
    public ArrayList<Tree<E>> getAllChildren(Tree<E> subtree, ArrayList<Tree<E>> list) {
        for (Tree<E> child : subtree.children) {
            list.add(child);
            getAllChildren(child, list);
        }
        return list;
    }

    // Works similar to the above method but sorts the list alphabetically and only works for Strings
    public ArrayList<Tree<String>> getAllChildrenStrings(Tree<String> subtree, ArrayList<Tree<String>> list) {
        subtree.children.sort(Comparator.comparing(child -> child.value));
        for (Tree<String> child : subtree.children) {
            list.add(child);
            getAllChildrenStrings(child, list);
        }
        return list;
    }

    // Public method used to get every parent of a specified subtree
    // The parent nodes are put into a list that is returned.
    public ArrayList<Tree<E>> getAllParents(Tree<E> subtree, ArrayList<Tree<E>> list) {
        if (subtree.parent != null) {
            list.add(subtree.parent);
            getAllParents(subtree.parent, list);
        }
        return list;
    }

    // Public method for printing the tree in a readable structure
    // Used for debugging only
    public void print(Tree<E> subtree) {
        for (int i = 0; i < subtree.depth; i++) System.out.print("\t");
        System.out.println(subtree.value);
        for (Tree<E> child : subtree.children) {
            print(child);
        }
    }
}