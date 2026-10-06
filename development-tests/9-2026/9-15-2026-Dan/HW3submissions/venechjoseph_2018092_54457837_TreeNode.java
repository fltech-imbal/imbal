
import java.util.ArrayList;

public class TreeNode {
    String data;
    TreeNode parent;
    ArrayList<TreeNode> children;

    public TreeNode(String data) {
        this.data = data;
        this.parent = null;
        this.children = new ArrayList<>();
    }
}