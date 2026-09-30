import java.io.File; // Needed to read files
import java.io.FileNotFoundException; // Needed to handle errors
import java.util.ArrayList; // Needed for text input
import java.util.Scanner; // Needed to use as a changable array for inputs

public class HW2_extra {
    
    private static int targetLength;
    
    private static String[] words;
    
    private static boolean[] used;

    public static void main(String[] args) throws FileNotFoundException {

        File inputFile = new File(args[0]);; // scanner to check the file

        ArrayList<String> list = new ArrayList<>(); // stores the words in the file

        try (Scanner scan = new Scanner(inputFile)) {
            targetLength = Integer.parseInt(scan.nextLine());

            // adds all to a string
            while (scan.hasNextLine()) {
                String word = scan.nextLine();
                if (!word.isEmpty()) {
                    list.add(word);
                }
            }
        } catch (FileNotFoundException e) {
            System.out.println("Error: file not found");
            e.printStackTrace();
            return;
        }

        // makes the list into an array
        words = list.toArray(new String[0]);
        used = new boolean[words.length];

        // finds the palendromes
        String[] phrase = new String[targetLength];
        findPalendromes(phrase); // realized that you can just make everything a static method to run the other methods in main
    }

    private static void findPalendromes(String[] startPhrase) {

        ArrayList<Integer> indexArrList = new ArrayList<>(); // Stores locations of indexes for later

        ArrayList<String[]> phraseArrList = new ArrayList<>(); // Stores phrases for later use

        ArrayList<boolean[]> usedArrList = new ArrayList<>(); // Stores wether a word has been used or not

        // Adds the starting values to the arrays
        indexArrList.add(0);
        phraseArrList.add(startPhrase.clone());
        usedArrList.add(used.clone());

        // Iterates through and finds permutations of the inital list of phrases
        while (!indexArrList.isEmpty()) {
            int index = indexArrList.remove(indexArrList.size() - 1);
            String[] current = phraseArrList.remove(phraseArrList.size() - 1);
            boolean[] currentUsed = usedArrList.remove(usedArrList.size() - 1);

            // Checks if it is a palendrome and then prints if is
            if (index == targetLength) {
                String s = String.join(" ", current);
                if (isPalindrome(s.replace(" ", ""))) {
                    System.out.println(s);
                }
                continue;
            }

            // generates a new phrase using the words in words
            for (int i = words.length - 1; i >= 0; i--) {
                // if the word has been used it dosent use it
                if (currentUsed[i]) {
                    continue;
                }
                
                boolean[] nextUsed = currentUsed.clone();
                String[] nextPhrase = current.clone();

                nextUsed[i] = true;
                nextPhrase[index] = words[i];

                indexArrList.add(index + 1);
                phraseArrList.add(nextPhrase);
                usedArrList.add(nextUsed);
            }
        }
    }

    // This is the non - recursive program that checks a string to see if it is a palindrome
    // It does this by checking if the first and last letters mathch and then moving first and last closer together
    private static boolean isPalindrome(String s) {
        int l = 0; // left index
        int r = s.length() - 1; // right index

        while (l < r) {
            if (s.charAt(l) != s.charAt(r)) {
                return false;
            }
            l++;
            r--;
        }
        return true;
    }
}
