ps -eaf | grep baselines*py | cut -d ' ' -f 4 | xargs kill
