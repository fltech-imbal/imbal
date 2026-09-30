import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

// Extra-credit version without recursion.
public class HW2Extra {
    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Usage: java HW2Extra <filename>");
            return;
        }

        List<String> words = new ArrayList<String>();
        int phraseLength;

        try {
            Scanner input = new Scanner(new File(args[0]));
            if (!input.hasNextInt()) {
                input.close();
                return;
            }
            phraseLength = input.nextInt();
            while (input.hasNext()) {
                words.add(input.next());
            }
            input.close();
        } catch (FileNotFoundException exception) {
            System.err.println("Cannot open file: " + args[0]);
            return;
        }

        if (phraseLength > words.size()) {
            return;
        }

        Collections.sort(words);
        int[] choices = new int[phraseLength];
        boolean[] used = new boolean[words.size()];
        int depth = 0;

        // Iteratively builds unique sorted phrases.
        while (depth >= 0) {
            if (choices[depth] == words.size()) {
                choices[depth] = 0;
                depth--;
                if (depth >= 0) {
                    used[choices[depth]] = false;
                    choices[depth]++;
                }
            } else if (used[choices[depth]]) {
                choices[depth]++;
            } else {
                used[choices[depth]] = true;
                if (depth == phraseLength - 1) {
                String candidate = buildPhrase(words, choices);
                if (isPalindrome(candidate)) {
                    System.out.println(candidate);
                }
                    used[choices[depth]] = false;
                    choices[depth]++;
                } else {
                    depth++;
                    choices[depth] = 0;
                }
            }
        }
    }

    // Builds one phrase.
    private static String buildPhrase(List<String> words, int[] choices) {
        StringBuilder result = new StringBuilder();
        for (int position = 0; position < choices.length; position++) {
            if (position > 0) {
                result.append(' ');
            }
            result.append(words.get(choices[position]));
        }
        return result.toString();
    }

    // Checks a phrase without recursion.
    private static boolean isPalindrome(String text) {
        int left = 0;
        int right = text.length() - 1;

        while (left < right) {
            if (!Character.isLetterOrDigit(text.charAt(left))) {
                left++;
            } else if (!Character.isLetterOrDigit(text.charAt(right))) {
                right--;
            } else if (Character.toLowerCase(text.charAt(left))
                    != Character.toLowerCase(text.charAt(right))) {
                return false;
            } else {
                left++;
                right--;
            }
        }
        return true;
    }
}
