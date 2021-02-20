

from selenium import webdriver
import time

host = 'http://127.0.0.1:5000/'
# set the driver path
driver_path = 'C:/Users/65909/Desktop/chromedriver'


def setup(path=driver_path, host=host):
    local_host = host
    web_driver = webdriver.Chrome(path)
    web_driver.get(local_host)
    return web_driver


def test_home_page(driver):
    time.sleep(2)
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(2)


def test_goto_feature_page(driver):
    # go to about_page from home_page
    about_button = driver.find_element_by_id('feat_button')
    about_button.click()
    time.sleep(2)


def test_goto_home_page(driver):
    # go back to home_page from about_page
    home_button = driver.find_element_by_id('home_button')
    home_button.click()
    time.sleep(2)


def test_goto_data_explanation_page(driver):
    # go to service-1 page from home page
    service = driver.find_element_by_id('service')
    service.click()
    time.sleep(1)
    service_1 = driver.find_element_by_id('data_ex')
    service_1.click()
    time.sleep(1)


def test_goto_modelexplanation_page(driver):
    # go to service-1 page from home page
    service = driver.find_element_by_id('service')
    service.click()
    time.sleep(1)
    service_2 = driver.find_element_by_id('model_ex')
    service_2.click()
    time.sleep(1)


browser = setup()


# test cases to check buttons and links
test_goto_home_page(driver=browser)
test_goto_feature_page(driver=browser)

test_goto_home_page(driver=browser)
test_goto_data_explanation_page(driver=browser)

test_goto_home_page(driver=browser)
test_goto_modelexplanation_page(driver=browser)

