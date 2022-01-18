package Selenium;

import io.github.bonigarcia.wdm.WebDriverManager;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class Trello {
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
    public void TrelloReviews() throws IOException {
        driver.get("https://www.g2.com/products/trello/reviews?");
        driver.manage().window().maximize();
        driver.manage().timeouts().implicitlyWait(5, TimeUnit.SECONDS);
        driver.findElement(By.xpath("//*[@id=\"new_user_consent\"]/input[6]")).click(); //cookie accept
        driver.findElement(By.xpath("//*[@id=\"survey-response-5173683\"]/div[2]/div[2]/div/div[2]/div/a/div[1]")).click(); //show more click

        // List<WebElement> reviews2 = driver.findElements(By.xpath("//*[@id='reviews']"));
        //List<WebElement> reviews2 = driver.findElements(By.xpath("//div[@class='nested-ajax-loading']"));
        //List<WebElement> reviews2 = driver.findElements(By.xpath("//div[@class='x-track-in-viewport-initialized']")); //retrieve name, date, review headline and comment (only one comment), retrieves most information
        List<WebElement> reviews2 = driver.findElements(By.xpath("//div[@class='paper__bd']")); //retrieve comment only (all comments), best format for reviews
        List<String> list = new ArrayList<>();

        for (WebElement webElement : reviews2) {
            String review = webElement.getText();
            System.out.println(review);
            list.add(review);


            FileWriter writer = new FileWriter("TrelloReviews.txt");
            for (String str : list) {
                writer.write(str + System.lineSeparator() + System.lineSeparator());
            }
            writer.close();
        }

    }
}