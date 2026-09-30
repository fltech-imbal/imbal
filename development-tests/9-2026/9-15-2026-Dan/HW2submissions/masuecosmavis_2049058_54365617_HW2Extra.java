import java.io.File;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Collections;

public class HW2Extra {
    
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Please type in a file name.");
            return;
        }
        
        int phrLen = 0;
        ArrayList<String> dict = new ArrayList<>();
        
        try {
            Scanner scan = new Scanner(new File(args[0]));
            if (scan.hasNextLine()) {
                phrLen = Integer.parseInt(scan.nextLine().trim());
            }
            while (scan.hasNextLine()) {
                String w = scan.nextLine().trim();
                if (!w.isEmpty()) {
                    dict.add(w);
                }
            }
            scan.close();
        } catch (Exception e) {
            System.out.println("Input file not found sadly.");
            return;
        }

        String[] words = dict.toArray(new String[0]);
        int numWords = words.length;
        ArrayList<String> palsFound = new ArrayList<>();
        
       
        int[] current = new int[phrLen]; 
        int[] loopind = new int[phrLen];
        boolean[] used = new boolean[numWords];
        
        int depth = 0;
        loopind[0] = 0; 
        
        while (depth >= 0) {
            
            if (depth == phrLen) {
                String combined = "";
                String strPhrase = "";
                
                for (int i = 0; i < phrLen; i++) {
                    combined += words[current[i]];
                    strPhrase += words[current[i]];
                    if (i < phrLen - 1) {
                        strPhrase += " ";
                    }
                }
                
                if (isaPal(combined)) {
                    palsFound.add(strPhrase);
                }
                
                depth--;
                if (depth >= 0) {
                    used[current[depth]] = false;
                    loopind[depth]++; 
                }
                continue;
            }
            
            if (loopind[depth] < numWords) {
                int i = loopind[depth];
                
                if (!used[i]) {
                    used[i] = true;
                    current[depth] = i; 
                    
                    if (depth + 1 < phrLen) {
                        loopind[depth + 1] = 0;
                    }
                    depth++; 
                } else {
                    loopind[depth]++;
                }
            } else {
                depth--;
                if (depth >= 0) {
                    used[current[depth]] = false;
                    loopind[depth]++;
                }
            }
        }
        
        Collections.sort(palsFound);
        for (int i = 0; i < palsFound.size(); i++) {
            System.out.println(palsFound.get(i));
        }
    }

    public static boolean isaPal(String str) {
        int left = 0;
        int right = str.length() - 1;
        
        while (left < right) {
            if (str.charAt(left) != str.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}