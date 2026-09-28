import java.io.File;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Collections;

public class HW2 {
    static int phrLength = 0;
    
    public static void main(String[] args) {
        if (args.length == 0) return;
        
        ArrayList<String> dict = new ArrayList<>();
        try {
            Scanner scan = new Scanner(new File(args[0]));
            if (scan.hasNextLine()) {
                phrLength = Integer.parseInt(scan.nextLine().trim());
            }
            while (scan.hasNextLine()) {
                String w = scan.nextLine().trim();
                if (!w.isEmpty()) dict.add(w);
            }
            scan.close();
        } catch (Exception e) {
            System.out.println("Input file not found!");
            return;
        }

        String[] wordarr = dict.toArray(new String[0]);
        boolean[] used = new boolean[wordarr.length];
        String[] current = new String[phrLength];
        ArrayList<String> palsfound = new ArrayList<>();
        
        
        findCombo(wordarr, used, current, 0, palsfound);
        
        Collections.sort(palsfound);
        for (int i = 0; i < palsfound.size(); i++) {
            System.out.println(palsfound.get(i));
        }
    }

    public static boolean checkforPal(String str, int left, int right) {
        if (left >= right) return true;
        if (str.charAt(left) != str.charAt(right)) return false;
        return checkforPal(str, left + 1, right - 1);
    }

    public static void findCombo(String[] words, boolean[] used, String[] current, int depth, ArrayList<String> results) {
        if (depth == phrLength) {
            String combined = "";
            String phraseStr = "";
            
            for (int i = 0; i < phrLength; i++) {
                combined += current[i];
                phraseStr += current[i] + (i == phrLength - 1 ? "" : " ");
            }
            
            if (checkforPal(combined, 0, combined.length() - 1)) {
                results.add(phraseStr);
            }
            return;
        }

        for (int i = 0; i < words.length; i++) {
            if (!used[i]) {
                used[i] = true;
                current[depth] = words[i];
                findCombo(words, used, current, depth + 1, results);
                used[i] = false; 
            }
        }
    }
}