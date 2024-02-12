# relevancy threshold: 0.5 (fixed)
# text query: apple, mug, cookies, table, bear
# model path: output/teatime_1, output/teatime_2, output/teatime_3

python render_text_query.py -m output/teatime_1 --text_query apple --rel_threshold 0.5
# python render_text_query.py -m output/teatime_2 --text_query apple --rel_threshold 0.5
# python render_text_query.py -m output/teatime_3 --text_query apple --rel_threshold 0.5

python render_text_query.py -m output/teatime_1 --text_query mug --rel_threshold 0.5
# python render_text_query.py -m output/teatime_2 --text_query mug --rel_threshold 0.5
# python render_text_query.py -m output/teatime_3 --text_query mug --rel_threshold 0.5

python render_text_query.py -m output/teatime_1 --text_query cookies --rel_threshold 0.5
# python render_text_query.py -m output/teatime_2 --text_query cookies --rel_threshold 0.5
# python render_text_query.py -m output/teatime_3 --text_query cookies --rel_threshold 0.5

python render_text_query.py -m output/teatime_1 --text_query table --rel_threshold 0.5
# python render_text_query.py -m output/teatime_2 --text_query table --rel_threshold 0.5
# python render_text_query.py -m output/teatime_3 --text_query table --rel_threshold 0.5

python render_text_query.py -m output/teatime_1 --text_query bear --rel_threshold 0.5
# python render_text_query.py -m output/teatime_2 --text_query bear --rel_threshold 0.5
# python render_text_query.py -m output/teatime_3 --text_query bear --rel_threshold 0.5