import java.util.Scanner;
import java.util.ArrayList;
import java.util.TreeSet;

public class HW2
{

    public static boolean isPalindrome(String str, int left, int right)
    {

        if (left >= right)
        {
            return true;
        }

        if (str.charAt(left) != str.charAt(right))
        {
            return false;
        }

        return isPalindrome(str, left + 1, right - 1);
    }

    public static boolean checkPalindrome(String str)
    {

        String clean = str.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();

        return isPalindrome(clean, 0, clean.length() - 1);
    }
    
    public static void findPalindromes(
            String[] words,
            boolean[] used,
            ArrayList<String> phrase,
            TreeSet<String> results)
    {

        if (phrase.size() >= 2)
        {
            String current = String.join(" ", phrase);

            if (checkPalindrome(current))
            {
                results.add(current);
            }
        }

        for (int i = 0; i < words.length; i++)
        {
            if (!used[i])
            {

                used[i] = true;
                phrase.add(words[i]);

                findPalindromes(words, used, phrase, results);

                phrase.remove(phrase.size() - 1);
                used[i] = false;
            }
        }
    }


    public static void main(String[] args)
    {
 
        Scanner input = new Scanner(System.in);
        ArrayList<String> wordList = new ArrayList<String>();

        while (input.hasNext())
        {
            wordList.add(input.next());
        }

        String[] words = wordList.toArray(new String[0]);
        boolean[] used = new boolean[words.length];

        ArrayList<String> phrase = new ArrayList<String>();
        TreeSet<String> results = new TreeSet<String>();

        findPalindromes(words, used, phrase, results);

        for (String result : results)
        {
            System.out.println(result);
        }

        input.close();
    }
}
