public class Entry{
    public String text;
    public int key;

    public Entry(String thisText, int thisKey){
        this.text = thisText;
        this.key = thisKey;
    }

    public String getText(){
        return this.text;
    }
    public int getKey(){
        return this.key;
    }
    
    public void setText(String newText){
        this.text = newText;
    }
    public void setKey(int newKey){
        this.key = newKey;
    }
}