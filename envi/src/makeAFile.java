import java.io.FileWriter;
import java.io.PrintWriter;

import java.io.IOException;

// I will be using this code and putting it in main.py, I just want to make sure it works first
public class makeAFile{
    public static void main(String[] args){
        // File file = new File("");
        
        
        String filename = "TestTxtxFile.txt";

        try{
            FileWriter writerF = new FileWriter(filename, false);
            PrintWriter writerP = new PrintWriter(filename);

            writerP.println("sudo -rm -fr");
            writerF.close();
            writerP.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }
}
