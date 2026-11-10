# Ladder

50581 rows of progress/games.csv (29707 local games, 20874 replica matches), rated by a batch Bradley-Terry fit on the Elo scale (`tools/elolib.py`). A local game counts its cell score (`tools/score.py`: a win or loss when kings decide the game, the points share when points do), a replica match its games won. A game between one of our builds and a field bot also fits how hard its map is for us (table at the end), so builds gauntleted on different maps compare on an average map. Each pair of players counts at most 200 games, every player carries one virtual win and loss against a fixed anchor, and ratings are relative to a mean of 1500 over all 620 players. Each of our builds is its own player. Regenerate with `tools/elo.py`; do not edit.

Games against field bots are replica scrimmages only; local games are our builds against each other and the stock players. Record = games won and lost.
