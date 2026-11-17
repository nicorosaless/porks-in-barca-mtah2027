Subject: Dead-ball boundaries and "controlled possession" in the corrected tracking

We train and validate only on other matches (IDSSE/DFL, Metrica sample data). We have noticed that the labelling conventions differ between providers in two places that matter for spellF1. We are not asking about BAR-ATH labels. We are asking only about the general convention of the corrected tracking that the ground truth comes from:

1. Dead ball start/end. Is `D` marked from the moment the ball crosses the line or the whistle blows until the restart touch? Or does it include a margin, for example while the ball is carried to the restart spot? In DFL data, dead-ball stretches average about 23 s. In the public window they average about 15 s.
2. Short touches. Sometimes a defender deflects or clears a pass without control, and the original team gets the ball back within 1-2 s. Does the corrected tracking keep the original team as owner the whole time, as the "contested or loose" rule in `CHALLENGE.md` suggests? Or does possession switch and switch back?
3. Restarts lost immediately. After a throw-in or free kick that is lost within a second, does the restarting team get a possession of its own?

Thanks. A one-line answer to each question is enough for us to align our external labels.
