package Selenium; // Generated by Selenium IDE
import io.github.bonigarcia.wdm.WebDriverManager;
import org.junit.Test;
import org.junit.Before;
import org.junit.After;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.Dimension;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.JavascriptExecutor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class JohnLewis {
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
    public void JohnLewisReviewsPg1() throws IOException {
        driver.get("https://www.fasterbroadband.co.uk/broadband-reviews/john-lewis?");
        driver.manage().window().maximize();
        driver.manage().timeouts().implicitlyWait(5, TimeUnit.SECONDS);

// make lists containing the WebElements
        List<String> ReviewList = new ArrayList<>();
        // List<String> NameList = new ArrayList<>();
        List<String> DateList = new ArrayList<>();
        List<String> ScoreList = new ArrayList<>();
// List containing all the WebElements
        List<String> AllList = new ArrayList<>();


        //reviews
        List<WebElement> reviews = driver.findElements(By.xpath("//div[@class='re-control-value']"));
        for (WebElement webElement : reviews) {
            String review = webElement.getText();
            //System.out.println(review);
            ReviewList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }

//        //Names
//        List<WebElement> names = driver.findElements(By.xpath("//span[@class='re-author-name']"));
//        for (WebElement webElement : names) {
//            String review = webElement.getText();
//            System.out.println(review);
//            NameList.add("\"" + review + "\"");
//            AllList.add("\"" + review + "\"");
//        }

        //Dates
        List<WebElement> dates = driver.findElements(By.xpath("//span[@class='re-date']"));
        for (WebElement webElement : dates) {
            String review = webElement.getText();
            //System.out.println(review);
            DateList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }

        //Scores
        List<WebElement> scores = driver.findElements(By.xpath("//div[@class='reviewBlock marked']"));
        String score, scoreXPath, final_score;
        List<WebElement> rating;
        int i;
        for (i = 0; i < scores.size(); i++) {
            score = scores.get(i).getAttribute("id");
            scoreXPath = "//div[@class='reviewBlock marked' and @id=" + score +"]";
            //rating = driver.findElements(By.xpath(scoreXPath + "//*[@class='re_header']/.."));
            //final_score = rating.get(i).getAttribute("data-rating");
            System.out.println(scoreXPath);
            ScoreList.add("\"" + scoreXPath + "\"");
            AllList.add("\"" + scoreXPath + "\"");
            // scoreXpath can iterate between the 6 reviews on the URL, need to get the score out of 5 attribute
        }

        System.out.println(ScoreList);

        //////////////////////////////////////////////////////////////////////
        FileWriter writer = new FileWriter("JlReviewsPg1.txt");
        for (String str : ScoreList) {
            writer.write(str + System.lineSeparator() + System.lineSeparator());
        }
        writer.close();

        File file = new File("JLPg1.csv");
        FileWriter fw = new FileWriter(file);
        BufferedWriter bw = new BufferedWriter(fw);
        bw.write("Review, Date, Score");
        bw.newLine();
        for (i = 0; i < (ReviewList.size()); i++) {
            bw.write(ReviewList.get(i) + ", " + DateList.get(i) + ", " + ScoreList.get(i));
            bw.newLine();
        }
        bw.close();
        fw.close();
    }

    @Test
    public void johnLewisReviewsPg2() throws IOException {
        driver.get("https://www.fasterbroadband.co.uk/broadband-reviews/john-lewis?jpage=2");
        driver.manage().window().setSize(new Dimension(1050, 660));
        driver.manage().timeouts().implicitlyWait(5, TimeUnit.SECONDS);
        driver.manage().window().maximize();

// make a list containing the 3 WebElements
        List<String> ReviewList = new ArrayList<>();
        List<String> NameList = new ArrayList<>();
        List<String> DateList = new ArrayList<>();

        List<String> AllList = new ArrayList<>();

        //reviews
        List<WebElement> reviews = driver.findElements(By.xpath("//div[@class='re-control-group _textarea']"));
        for (WebElement webElement : reviews) {
            String review = webElement.getText();
            System.out.println(review);
            ReviewList.add(review);
            AllList.add(review);
        }
        //Names
        List<WebElement> names = driver.findElements(By.xpath("//span[@class='re-author-name']"));
        for (WebElement webElement : names) {
            String review = webElement.getText();
            System.out.println(review);
            NameList.add(review);
            AllList.add(review);
        }

        //Dates
        List<WebElement> dates = driver.findElements(By.xpath("//span[@class='re-date']"));
        for (WebElement webElement : dates) {
            String review = webElement.getText();
            System.out.println(review);
            DateList.add(review);
            AllList.add(review);
        }

        System.out.println(AllList);

        FileWriter writer = new FileWriter("JlReviewsPg2.txt");
        for (String str : AllList) {
            writer.write(str + System.lineSeparator() + System.lineSeparator());
        }
        writer.close();

        File file = new File("JLPg2.csv");
        FileWriter fw = new FileWriter(file);
        BufferedWriter bw = new BufferedWriter(fw);
        bw.write("Name, Date, Review");
        bw.newLine();
        for(int i=0;i<(ReviewList.size());i++)
        {
            bw.write(NameList.get(i)+", "+DateList.get(i)+", "+ReviewList.get(i));
            bw.newLine();
        }
        bw.close();
        fw.close();
    }

    @Test
    public void johnLewisReviewsPg3() throws IOException {
        driver.get("https://www.fasterbroadband.co.uk/broadband-reviews/john-lewis?jpage=3");
        driver.manage().window().setSize(new Dimension(1050, 660));
        driver.manage().timeouts().implicitlyWait(5, TimeUnit.SECONDS);
        driver.manage().window().maximize();

// make a list containing the 3 WebElements
        List<String> ReviewList = new ArrayList<>();
        List<String> NameList = new ArrayList<>();
        List<String> DateList = new ArrayList<>();
        List<String> RatingList = new ArrayList<>();

        List<String> AllList = new ArrayList<>();

        //reviews
        List<WebElement> reviews = driver.findElements(By.xpath("//div[@class='re-control-group _textarea']"));
        for (WebElement webElement : reviews) {
            String review = webElement.getText();
            System.out.println(review);
            ReviewList.add(review);
            AllList.add(review);
        }

        //Names
        List<WebElement> names = driver.findElements(By.xpath("//span[@class='re-author-name']"));
        for (WebElement webElement : names) {
            String review = webElement.getText();
            System.out.println(review);
            NameList.add(review);
            AllList.add(review);
        }

        //Dates
        List<WebElement> dates = driver.findElements(By.xpath("//span[@class='re-date']"));
        for (WebElement webElement : dates) {
            String review = webElement.getText();
            System.out.println(review);
            DateList.add(review);
            AllList.add(review);
        }

        System.out.println(AllList);

        FileWriter writer = new FileWriter("JlReviewsPg3.txt");
        for (String str : AllList) {
            writer.write(str + System.lineSeparator() + System.lineSeparator());
        }
        writer.close();

        File file = new File("JLPg3.csv");
        FileWriter fw = new FileWriter(file);
        BufferedWriter bw = new BufferedWriter(fw);
        bw.write("Name, Date, Review");
        bw.newLine();
        for(int i=0;i<(ReviewList.size());i++)
        {
//            bw.write(NameList.get(i)+", "+DateList.get(i)+", "+ReviewList.get(i));
            bw.write(NameList.get(i)+", "+DateList.get(i)+", "+ReviewList.get(i));
            bw.newLine();
        }
        bw.close();
        fw.close();
    }


    @Test
    public void johnLewisReviewsAllPages() throws IOException {

       // make a list containing the 3 WebElements
        List<String> ReviewList = new ArrayList<>();
        List<String> NameList = new ArrayList<>();
        List<String> DateList = new ArrayList<>();

        List<String>AllList = new ArrayList<>();

        driver.get("https://www.fasterbroadband.co.uk/broadband-reviews/john-lewis?");
        driver.manage().window().setSize(new Dimension(1050, 660));
        driver.manage().timeouts().implicitlyWait(5, TimeUnit.SECONDS);
        driver.manage().window().maximize();

        //reviews
        List<WebElement> reviews = driver.findElements(By.xpath("//div[@class='re-control-value']"));
        for (WebElement webElement : reviews) {
            String review = webElement.getText().replaceAll(",","");
            System.out.println(review);
            ReviewList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }

        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        //Names
        List<WebElement> names = driver.findElements(By.xpath("//span[@class='re-author-name']"));
        for (WebElement webElement : names) {
            String review = webElement.getText();
            System.out.println(review);
            NameList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }

        try {
            Thread.sleep( 1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        //Dates
        List<WebElement> dates = driver.findElements(By.xpath("//span[@class='re-date']"));
        for (WebElement webElement : dates) {
            String review = webElement.getText();
            System.out.println(review);
            DateList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }

        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        ////////////////////////page 2 ////////////////////////////////
       driver.get("https://www.fasterbroadband.co.uk/broadband-reviews/john-lewis?jpage=2");


        //reviews
        List<WebElement> reviews2 = driver.findElements(By.xpath("//div[@class='re-control-value']"));
        for (WebElement webElement : reviews2) {
            String review = webElement.getText().replaceAll(",","");
            System.out.println(review);
            ReviewList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }

        //Names
        List<WebElement> names2 = driver.findElements(By.xpath("//span[@class='re-author-name']"));
        for (WebElement webElement : names2) {
            String review = webElement.getText();
            System.out.println(review);
            NameList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }

        //Dates
        List<WebElement> dates2 = driver.findElements(By.xpath("//span[@class='re-date']"));
        for (WebElement webElement : dates2) {
            String review = webElement.getText();
            System.out.println(review);
            DateList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }


        ////////////////////////page 3 ////////////////////////////////
        driver.get("https://www.fasterbroadband.co.uk/broadband-reviews/john-lewis?jpage=3");

        //reviews
        List<WebElement> reviews3 = driver.findElements(By.xpath("//div[@class='re-control-value']"));
        for (WebElement webElement : reviews3) {
            String review = webElement.getText().replaceAll(",","");
            System.out.println(review);
            ReviewList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }



        //Names
        List<WebElement> names3 = driver.findElements(By.xpath("//span[@class='re-author-name']"));
        for (WebElement webElement : names3) {
            String review = webElement.getText();
            System.out.println(review);
            NameList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }

        //Dates
        List<WebElement> dates3 = driver.findElements(By.xpath("//span[@class='re-date']"));
        for (WebElement webElement : dates3) {
            String review = webElement.getText();
            System.out.println(review);
            DateList.add("\"" + review + "\"");
            AllList.add("\"" + review + "\"");
        }


        System.out.println(AllList);


        FileWriter writer = new FileWriter("JlAllReviews.txt");
        for (String str : AllList) {
            writer.write(str + System.lineSeparator() + System.lineSeparator());
        }
        writer.close();

        File file = new File("JLAllPages.csv");
        FileWriter fw = new FileWriter(file);
        BufferedWriter bw = new BufferedWriter(fw);
        bw.write("Review,Date");
        bw.newLine();
        for(int i=0;i<(ReviewList.size());i++)
        {
            bw.write(ReviewList.get(i)  + ", " + DateList.get(i));
            bw.newLine();
        }
        bw.close();
        fw.close();
    }

}


// //ul[@class="jreview-pagination"]/li use this xPath, get href and itterate


//href="https://www.fasterbroadband.co.uk/broadband-reviews/john-lewis?jpage=2"  page 2 href
//*[@id="jreview-pagination"]/ul/li[2]/a    page 2 xpath
//href="https://www.fasterbroadband.co.uk/broadband-reviews/john-lewis?jpage=3"  page 3 href
//*[@id="jreview-pagination"]/ul/li[3]/a    page 3 xpath


//driver.findElement(By.xpath("//*[@id=\"jreview-pagination\"]/ul/li[2]/a")).click();