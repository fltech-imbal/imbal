import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

// Finds recursive palindrome phrases.
public class HW2 {
    // Reads input and prints matches.
    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Usage: java HW2 <filename>");
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

        Collections.sort(words);
        boolean[] used = new boolean[words.size()];
        findPalindromes(words, phraseLength, used, new ArrayList<String>());
    }

    // Recursively builds unique sorted phrases.
    private static void findPalindromes(List<String> words, int phraseLength,
                                        boolean[] used, List<String> phrase) {
        if (phrase.size() == phraseLength) {
            String candidate = joinPhrase(phrase);
            if (isPalindrome(candidate, 0, candidate.length() - 1)) {
                System.out.println(candidate);
            }
            return;
        }

        for (int index = 0; index < words.size(); index++) {
            if (!used[index]) {
                used[index] = true;
                phrase.add(words.get(index));
                findPalindromes(words, phraseLength, used, phrase);
                phrase.remove(phrase.size() - 1);
                used[index] = false;
            }
        }
    }

    // Joins words with spaces.
    private static String joinPhrase(List<String> phrase) {
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < phrase.size(); index++) {
            if (index > 0) {
                result.append(' ');
            }
            result.append(phrase.get(index));
        }
        return result.toString();
    }

    // Recursively checks a palindrome.
    private static boolean isPalindrome(String text, int left, int right) {
        if (left >= right) {
            return true;
        }
        if (!Character.isLetterOrDigit(text.charAt(left))) {
            return isPalindrome(text, left + 1, right);
        }
        if (!Character.isLetterOrDigit(text.charAt(right))) {
            return isPalindrome(text, left, right - 1);
        }
        if (Character.toLowerCase(text.charAt(left))
                != Character.toLowerCase(text.charAt(right))) {
            return false;
        }
        return isPalindrome(text, left + 1, right - 1);
    }
}
