.PHONY: format
format:
	@yapf --in-place mmlib.py spatial.py server.py \
		fetch_map.py graphical_models.py observability.py

.PHONY: profile
profile:
	@python3 -m cProfile -s cumulative -o mmlib.prof \
		mmlib.py --input-filename san-francisco.graphml \
		--input-trace ../data/trace.txt > output.txt
	@python3 -m flameprof mmlib.prof > mmlib.profile.svg


.PHONY: server
server:
	@FLASK_ENV=development python3 server.py
