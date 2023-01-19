import argparse
import datetime as dt
import logging
import re
import string
import time
import numpy as np
import pandas as pd
from psaw import PushshiftAPI

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_text(text):
	text = re.sub(r'\s{1,}', ' ', text)
	remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
	text = text.translate(remove_punct_map).lower()
	return text


def make_reddit_dataset_files(subreddits_no, offset, subreddit_size, output_path):
	subreddits_df = pd.read_csv('data/subreddit_info.csv',
		usecols=['subreddit', 'in_data'])
	subreddits = subreddits_df[
			subreddits_df['in_data'] == True
		].iloc[offset:(offset + subreddits_no)]['subreddit']

	# Map each subreddit in the list to its index
	labels_map = {subreddit: (i+offset) for i, subreddit in enumerate(subreddits)}

	# Call API to get submissions for each subreddit
	api = PushshiftAPI()
	data = []

	for subreddit in subreddits:
		# We can't query too many submissions at once. Rather use 'created_utc'
		# field and the 'before' search parameter to get to the desired number
		retrieved = 0
		epoch=int(dt.datetime(2022, 1, 1).timestamp())
		before = len(data)
		while retrieved < subreddit_size:
			gen = api.search_submissions(
				subreddit=subreddit,
				fields=['subreddit', 'title', 'selftext', 'created_utc'],
				limit=min(500, subreddit_size - retrieved),
				user_removed=False,
				mod_removed=False,
				before=epoch)
			for thing in gen:
				data.append([
					process_text(thing.d_['title'] + " " + thing.d_.get('selftext', '')),
					thing.d_['subreddit'],
					labels_map[thing.d_['subreddit']]])
				epoch = min(epoch, thing.d_['created_utc'])
			retrieved += 500
		after = len(data)
		logger.info(f'Fetched {after-before} from subreddit {subreddit}')

	data = np.array(data)
	np.random.shuffle(data)

	posts_df = pd.DataFrame(data)
	posts_df.columns = ['text', 'subreddit', 'label']
	
	posts_df.to_csv(output_path)


def setup_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='Use Pushshift API to download Reddit data')

	parser.add_argument('--subreddits', type=int, help='number of subreddits', default=10)
	parser.add_argument('--offset', type=int, default=0,
		help='number of subreddits to skip (useful when creating dataset for new task)')
	parser.add_argument('--size', type=int, help='size of subreddits', default=1000)
	parser.add_argument('--output', type=str, default='dataset/reddit_posts.csv',
		help='output path for dataset (CSV)')

	return parser


def main():
	parser = setup_argparser()
    args = parser.parse_args()

	start = time.time()
	make_reddit_dataset_files(args.subreddits, args.offset, args.size, args.output)
	elapsed = time.time() - start

	logger.info(f'Fetched {args.subreddits * args.size} posts in {elapsed} seconds')


if __name__ == '__main__':
	main()