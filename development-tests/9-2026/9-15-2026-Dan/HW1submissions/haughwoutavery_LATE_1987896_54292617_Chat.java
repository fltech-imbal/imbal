
/**
 * Chat: Holds customer, representative, and time initiated together :) 
 * Avery Haughwout
 * CSE2012
 * 8/27/26
 */

public class Chat{
    static Request curReq;
    static String curRep;
    public Chat(Request request, String rep){
        this.curReq = request;
        this.curRep = rep;
    }
    public Request getReq(){
        return this.curReq;
    }
    public String getRep(){
        return this.curRep;
    }
}
