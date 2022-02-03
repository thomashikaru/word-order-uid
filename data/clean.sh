cat $1 | sed "/^[[:space:]]=/d" | sed '/^[[:space:]]*$/d'
