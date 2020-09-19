from key import TwitterKey
import tweepy
import random


class ElonOrKanye:
    def __init__(self):
        
        consumer_key = ""
        consumer_secret = ""
        access_token_key = ""
        access_token_secret = ""
        
        key = TwitterKey()  # create your own key singleton
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token_key, access_token_secret)

        print("getting tweets...")
        self.api = tweepy.API(auth)
        print("getting Elon's tweets...")
        self.elon_texts = self.fetch_tweets('@elonmusk')
        print("getting Kanye's tweets...")
        self.kanye_texts = self.fetch_tweets('@kanyewest')

    # helper function that gets past 3200(maximum) tweets of screen_name
    def fetch_tweets(self, screen_name):
        timelines = tweepy.Cursor(self.api.user_timeline,
                                  screen_name=screen_name,
                                  tweet_mode="extended",
                                  include_rts=False,
                                  exclude_replies=True).items()
        text_list = []
        for status in timelines:
            if not (("http://" in status.full_text) or "https://" in status.full_text):
                text_list.append(status.full_text)
        return text_list

    def get_kanye(self):
        text = self.kanye_texts.pop(random.randint(0, len(self.kanye_texts)))
        return text, "kanye"

    def get_elon(self):
        text = self.elon_texts.pop(random.randint(0, len(self.elon_texts)))
        return text, "elon"

    def get_question(self):
        rand = random.random()
        if rand < 0.5:
            text, answer = self.get_kanye()
        else:
            text, answer = self.get_elon()
        return text, answer


if __name__ == "__main__":
    game = ElonOrKanye()
    total = 0
    correct = 0
    while len(game.elon_texts) > 0 and len(game.kanye_texts) > 0:
        text, answer = game.get_question()
        print("--------Round" + str(int(total)) + "--------")
        print(text)
        user_input = input("Who tweeted this?: ").lower().strip(' ')
        if answer == user_input:
            print("correct!")
            correct += 1
        else:
            print("nope. It was " + answer)
        total += 1
        if total % 10 == 0:
            print("your current accuracy:" + str(round(correct / total, 2)))
        print()
    print("Game Over. Your final accuracy:" + str(round(correct / total, 2)))
