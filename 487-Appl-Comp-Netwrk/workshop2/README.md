# workshop-2-python-chat-egirgin

**Student ID:** 2016400099

**Name&Surname:** Emre Girgin

**Tested classmate:** Alperen Bağ

**Python3.6**

**Requirements:** There is not any requirement. You only need to have _netcat_ and _python3_. The libraries I used are _socket_, _subprocess_, _json_, _os_, _enum_, _sys_, and _shutil_ , which are exists in standard library. 

**Running** Please remember that you need to have an alias for python such that both _python_ and _python3_ commands should refer to Python3. Creating a virtual env with python3 will solve this issue by default. 

**Command** ```python chatApp.py``` or ```python3 chatApp.py``` 

**Runtime** 2 keywords _scan_ and _quit_ allow you to explore program. There is not such system that checking regularly for new packets, rather you type _scan_ to refresh. You can type _scan_ to see all the users logged and all the new messages came from a user. Similarly, you can use _quit_ to turn back main page, where you see all the users you can chat, or _quit_ the program safely in the main menu.

**Example case**

1-) Run app and enter your username.

2-) Type _scan_ until you see the username you want to chat with.

3-) Type the target username

4-) Type the message you want to send.

5-) Type _scan_ to see if there is new messages.

6-) Type _quit_ to turn back to main menu.

7-) Type _quit_ to close the app safely.


-> Everytime the app restarted, all the saved data has gone. If you want to keep, comment _clearLogs()_ function. Then, all the messages and username - ip pairs restored back.

-> Tested on both Ubuntu 18.04 and 16.04

