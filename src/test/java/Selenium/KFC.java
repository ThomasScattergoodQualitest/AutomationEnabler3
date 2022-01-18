package Selenium;

import io.github.bonigarcia.wdm.WebDriverManager;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class KFC {
    private WebDriver driver;
    private Map<String, Object> vars;
    JavascriptExecutor js;

    @Before
    public void setUp() {
        WebDriverManager.chromedriver().setup();
        driver = new ChromeDriver();
        js = (JavascriptExecutor) driver;
        vars = new HashMap<String, Object>();
    }

    @After
    public void tearDown() {
        driver.quit();
    }

    @Test
    public void KFCFacebookReviews() throws IOException, InterruptedException {
        List<String> fileContent = new ArrayList<>();
        driver.get("https://www.facebook.com/kfc/reviews/");
        driver.manage().window().maximize();
        driver.manage().timeouts().implicitlyWait(2, TimeUnit.SECONDS);
        driver.findElement(By.xpath("//button[text()='Allow All Cookies']")).click(); //accept cookies
        JavascriptExecutor js = (JavascriptExecutor) driver;
        List<String> list = new ArrayList<>();
        js.executeScript("window.scrollTo(0, document.body.scrollHeight);");
        List<WebElement> reviews2 = null;
        try {
            long lastHeight=(long)js.executeScript("return document.body.scrollHeight");
            int scrollCount = 0;
            while (scrollCount<6) {
                reviews2 = driver.findElements(By.xpath("(//div[contains(@class,'userContentWrapper')])"));
                // System.out.println(reviews2);
                js.executeScript("window.scrollTo(0, document.body.scrollHeight);");
                Thread.sleep(2000);
                long newHeight = (long)js.executeScript("return document.body.scrollHeight");
                if (newHeight == lastHeight) {
                    break;
                }
                lastHeight = newHeight;
                scrollCount++;
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        for (int i = 0; i < reviews2.size(); i++) {
            String date = reviews2.get(i).findElements(By.xpath("(//span[@class='timestampContent'])")).get(i).getText().replaceAll("," ,"");
            String review = reviews2.get(i).findElements(By.xpath("//div[@data-testid='post_message']")).get(i).getText().replaceAll(",","");

            String xpath = review.substring(0,9);

            List <WebElement> likes = driver.findElements(By.xpath("//div[@data-testid='post_message']//p[contains(text(),'"+xpath+"')]" + "/parent::div/../../../../parent::div//span[@aria-label='See who reacted to this']/span/a"));
//            StringBuilder like = new StringBuilder();
            List<String> like = new ArrayList<>();
            like.add("0");
            like.add("0");
            like.add("0");

            for(int j=0;j< likes.size();j++){
                String reaction = likes.get(j).getAttribute("aria-label");
                if(reaction.contains("Like")){
                    like.set(0,("\"" + reaction + "\"").replaceAll(" Like",""));
                }
                if(reaction.contains("Haha")){
                    like.set(1,(reaction).replaceAll(" Haha",""));
                }
                if(reaction.contains("Angry")){
                    like.set(2,(reaction).replaceAll(" Angry",""));
                }
            }
            System.out.println(like);
            if(like.toString().equals("")) {
                fileContent.add("\"" + review + "\"" + "," + "\"0\"" + "," + "\"" + date + "\"");
            }
            else{
                fileContent.add("\"" + review + "\"" + ","  + like.toString().replace("[", "").replace("]", "")+ ","   + "\"" + date + "\"");
            }
        }
        File file = new File("KFCReviews.csv");
        FileWriter fw = new FileWriter(file);
        BufferedWriter bw = new BufferedWriter(fw);
        bw.write("Review,Like,Haha,Angry,Date");
        bw.newLine();
        for (String s : fileContent) {
            bw.write(s);
            bw.newLine();
        }
        bw.close();
        fw.close();
    }
}