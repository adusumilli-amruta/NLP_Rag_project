css = '''
<style>
/* Chat message container styling */
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* User message styling */
.chat-message.user {
    background: linear-gradient(135deg, #4a90e2, #357ABD);
    color: white;
    margin-left: auto;
    margin-right: 15%;
    max-width: 70%;
    text-align: left;
}

/* Bot message styling */
.chat-message.bot {
    background: linear-gradient(135deg, #475063, #2b313e);
    color: white;
    margin-right: auto;
    margin-left: 15%;
    max-width: 70%;
    text-align: left;
}

/* Avatar container */
.chat-message .avatar {
    width: 60px;
    height: 60px;
    flex-shrink: 0;
    overflow: hidden;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    background: white; /* Add background for better visuals */
}

/* Avatar image */
.chat-message .avatar img {
    width: 100%;
    height: 100%;
    object-fit: contain; /* Ensures the image fits without distortion */
    border-radius: 50%;
}

/* Message container */
.chat-message .message {
    font-size: 1rem;
    line-height: 1.5;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.1);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Add hover effect for messages */
.chat-message:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}

/* Adjust margin for the messages */
.chat-message.user {
    margin-right: 10%;
}

.chat-message.bot {
    margin-left: 10%;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712104.png" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''



user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/706/706830.png" alt="User Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
