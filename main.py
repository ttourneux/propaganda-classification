#import all_data_loop
#import twitter_loop
#import non_twitter_loop
import data_loop
import results

def main():

    data_loop.data_loop('twitter')## "twitter" or "all_data" or "non_twitter"
    
    #all_data_loop.all_data_loop()
    #twitter_loop.twitter_loop()
    #non_twitter_loop.non_twitter_loop()
    
    
    
    ## feature selection implementation 
    
    
    #results.plot_accuracy()
main()


