# 🧑‍🍳 WAIter: Voice Chatbot for Restaurants

WAIter is an interactive chatbot with voice capabilities designed for restaurant scenarios. It simulates **Miguel**, a friendly and helpful Cadiz waiter who can respond to customers' spoken questions in real-time, take orders, and send them directly to the kitchen.

**🗣️ Note: The voice assistant operates in Spanish** - Miguel speaks authentic Andalusian Spanish, making it perfect for Spanish-speaking restaurants and customers who want an authentic Spanish dining experience.

## 🚀 [**TRY THE LIVE DEMO**](https://huggingface.co/spaces/ItzRoBeerT/WAIter?logs=container)

You can try WAIter right now without installation on our HuggingFace Space. You just need a microphone to talk to Miguel in Spanish!

---

## 🛠️ Tech Stack

### **Voice and Audio Processing**
- 🎙️ **[FastRTC](https://fastrtc.org)** - Real-time audio streaming with Voice Activity Detection (VAD)
- 🔊 **[ElevenLabs](https://elevenlabs.io/)** - Expressive and natural voice synthesis (TTS)
- 🗣️ **[Whisper](https://openai.com/research/whisper)** (via Groq) - High-precision speech recognition

### **Artificial Intelligence**
- 💬 **[LangChain](https://www.langchain.com/)** - Advanced LLM orchestration
- 🔄 **[LangGraph](https://github.com/langchain-ai/langgraph)** - Complex workflows with agents
- 🧠 **[Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/)** (via OpenRouter) - Main language model
- 🔍 **[HuggingFace Embeddings](https://huggingface.co/)** (BAAI/bge-m3) - Vector semantic search

### **Data Management**
- 💾 **Markdown Knowledge Base** - Complete restaurant menu
- 🗄️ **[Supabase](https://supabase.com/)** - Real-time order management database
- 📊 **RAG (Retrieval Augmented Generation)** - Intelligent menu search

### **Interface and Deployment**
- 🧪 **[Gradio](https://www.gradio.app/)** - Interactive web interface with voice support
- ☁️ **HuggingFace Spaces** - Deployment and live demo

---

## ✨ Key Features

### 🎯 **User Experience**
- **Natural voice interaction** - Talk directly to Miguel as if he were a real waiter (in Spanish)
- **Text and audio responses** - Choose how you want to interact
- **Authentic personality** - Miguel is a Cadiz waiter with charm and professionalism
- **Conversational memory** - Remembers your order throughout the entire conversation

### 🍽️ **Restaurant Functionality**
- **Complete menu consultation** - Detailed information about all dishes
- **Intelligent search** - Find dishes by ingredients, prices, or dietary preferences
- **Allergen management** - Complete information about allergens and special dietary options
- **Complete order taking** - From consultation to sending to kitchen
- **Dietary options** - Vegetarian, vegan, gluten-free, and diabetic menus

### 🔧 **Advanced Technology**
- **Real-time processing** - Immediate responses without perceptible latency
- **Intelligent RAG** - Vector search to find relevant menu information
- **Kitchen integration** - Orders are automatically sent to the database
- **Specialized tool system** - Specific tools for menu consultation and order sending

---

## 🚀 Getting Started

### **Prerequisites**
- Python 3.10 or higher
- API keys for required services (see below)
- Microphone (for voice functionality)
- **Basic Spanish understanding** (for voice interaction with Miguel)

### **1. Clone and Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/waiter-chatbot.git
cd waiter-chatbot

# Create a virtual environment (recommended)
python -m venv waiter-env
source waiter-env/bin/activate  # On Windows: waiter-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Environment Variables Configuration**

Create a `.env` file in the project root with your API keys:

```env
# REQUIRED - For the language model
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# REQUIRED for voice functionality - Transcription
GROQ_API_KEY=your_groq_key

# REQUIRED for voice functionality - Voice synthesis
ELEVENLABS_API_KEY=your_elevenlabs_key

# OPTIONAL - For advanced logging
HELICONE_API_KEY=your_optional_helicone_key

# REQUIRED for sending orders to kitchen
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### **3. Getting API Keys**

#### **OpenRouter (REQUIRED)**
1. Go to [OpenRouter](https://openrouter.ai/)
2. Create an account and get your API key
3. Make sure you have credits to use Gemini 2.5 Flash

#### **Groq (REQUIRED for voice)**
1. Go to [Groq](https://console.groq.com/)
2. Create a free account
3. Get your API key to use Whisper

#### **ElevenLabs (REQUIRED for voice)**
1. Go to [ElevenLabs](https://elevenlabs.io/)
2. Create an account (free plan available)
3. Get your API key

#### **Supabase (REQUIRED for orders)**
1. Go to [Supabase](https://supabase.com/)
2. Create a new project
3. Set up the necessary tables (see Database Setup section)
4. Get your project URL and API key

### **4. Supabase Database Setup**

Run these SQL commands in the Supabase SQL editor:

```sql
-- Orders table
CREATE TABLE orders (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    order_id TEXT NOT NULL,
    table_number TEXT NOT NULL,
    special_instructions TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Order items table
CREATE TABLE order_items (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    order_id BIGINT REFERENCES orders(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    quantity INTEGER DEFAULT 1,
    variations TEXT DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for better performance
CREATE INDEX idx_orders_order_id ON orders(order_id);
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
```

### **5. Running the Application**

```bash
# Run the application
python app.py

# The application will automatically open in your browser
# Default at: http://localhost:7860
```

---

## 📂 Project Structure

```
waiter-chatbot/
├── app.py                 # 🎯 Main application with Gradio interface
├── agent.py              # 🤖 LangGraph agent for Miguel the waiter
├── model.py              # 🧠 Language model manager
├── tools.py              # 🔧 Specialized tools (menu, orders)
├── supabase_client.py    # 🗄️ Client for Supabase operations
├── requirements.txt      # 📦 Project dependencies
├── .env                  # 🔑 Environment variables (create manually)
├── .gitignore           # 🚫 Files ignored by git
│
├── data/
│   ├── carta.md         # 🍽️ Complete menu (knowledge base)
│   └── system_prompt.txt # 👨‍🍳 System prompt for Miguel (in Spanish)
│
└── utils/
    ├── classes.py       # 📋 Data classes (Order, etc.)
    ├── logger.py        # 📝 Colored logging system
    └── functions.py     # 🛠️ Auxiliary functions
```

---

## 🎭 Meet Miguel - Your Virtual Waiter

**Miguel** is an authentic Cadiz waiter working at a traditional restaurant in Cadiz specializing in traditional Andalusian cuisine.

### **Miguel's Personality:**
- 🏛️ **Authentic Cadiz native** - With genuine expressions and local idioms
- 👔 **Professional but friendly** - Knows how to balance efficiency with warmth
- 💡 **Proactive** - Suggests dishes and anticipates your needs
- 🧠 **Good memory** - Remembers your order throughout the conversation
- ⚡ **Efficient** - Optimized for natural and fluid voice responses
- 🇪🇸 **Spanish speaker** - Communicates in authentic Andalusian Spanish

---

## 🔄 How the System Works

### **Voice Conversation Flow:**

1. **🎤 Audio Capture** → FastRTC captures your voice in real-time
2. **📝 Transcription** → Groq (Whisper) converts audio to text
3. **🧠 Processing** → Miguel's LangGraph agent processes your request
4. **🔍 RAG Search** → Searches for relevant information in menu
5. **⚙️ Tools** → Uses specialized tools as needed:
   - `restaurant_menu_lookup_tool` - For menu queries
   - `send_order_to_kitchen_tool` - For sending orders
6. **💬 Generation** → Gemini 2.5 generates contextual response as Miguel
7. **🔊 Synthesis** → ElevenLabs converts response to natural voice
8. **📱 Interface** → Gradio displays conversation and plays audio

### **Text Conversation Flow:**

Same process but without audio capture and voice synthesis steps, ideal for users who prefer typing or in environments where audio cannot be used.

---

## 🍽️ Menu Capabilities

### **Available Sections:**
- 🐟 **Bay and Almadraba Preserves** - Tuna, mojama, anchovies
- 🍤 **Cadiz Tapas** - Shrimp fritters, marinated dogfish, sea anemones
- 🥘 **Hearty Stews** - Cadiz cabbage stew, legume dishes, oxtail in sherry
- 🍤 **Fried Fish and Seafood** - Mixed fried fish, prawns, shrimp
- 🥩 **Local Meats** - Cadiz retinto beef, fighting bull meat
- 🍚 **Bay Rice Dishes** - With lobster, cuttlefish, almadraba tuna
- 🍮 **Cadiz Desserts** - Tocino de cielo, bienmesabe, pestiños
- 🍷 **Beverages** - Sherry wines, manzanilla, rebujito

### **Dietary Options:**
- 🌱 **Vegetarian and Vegan** - Complete adapted menus
- 🌾 **Gluten-Free** - Preparation in contamination-free zone
- 💉 **For Diabetics** - Options without added sugars
- 👶 **Children's Menu** - Adapted portions and flavors

### **Allergen Information:**
Miguel can consult detailed information about all major allergens and help you find safe options according to your needs.

---

## 🏗️ Use Cases

### **For Restaurants:**
- **Order taking automation** during peak hours
- **24/7 information** about menu and allergens
- **Staff workload reduction**
- **Consistent experience** independent of human waiter

### **For Customers:**
- **Natural menu exploration** without pressure
- **Detailed information** about ingredients and preparation
- **Personalized recommendations** based on preferences
- **Accessibility** for people with reading difficulties
- **Spanish language practice** for learners

### **For Developers:**
- **Complete template** for restaurant chatbots
- **Multiple API integration** for voice and AI
- **Specialized RAG system** for hospitality
- **Scalable architecture** with LangGraph

---

## 🔧 Advanced Configuration

### **Model Customization:**

You can change the language model in `model.py`:

```python
# Available models in OpenRouter
models = [
    "google/gemini-2.5-flash-preview",      # Fast and efficient (recommended)
    "anthropic/claude-3.5-sonnet",          # High quality
    "openai/gpt-4o",                        # Robust alternative
]
```

### **Waiter Customization:**

Edit `data/system_prompt.txt` to change:
- Waiter **personality**
- Restaurant **name**
- Conversation **style**
- Specific behavior **rules**
- **Language** (currently optimized for Spanish)

### **Menu Customization:**

Edit `data/carta.md` to:
- **Change dishes** and prices
- **Add new sections**
- **Modify restaurant information**
- **Adapt to your business**

### **Voice Configuration:**

In `app.py` you can adjust:

```python
# ElevenLabs configuration
voice_settings = VoiceSettings(
    stability=0.0,        # Voice variability
    similarity_boost=1.0, # Similarity to base voice
    style=0.0,           # Expressive style
    use_speaker_boost=True, # Clarity enhancement
    speed=1.1,           # Speech speed
)
```

---

## 🐛 Troubleshooting

### **Common Issues:**

#### **Error: "No API key found"**
- ✅ Verify that the `.env` file is in the project root
- ✅ Check that variables are written correctly
- ✅ Restart the application after adding keys

#### **Audio error: "Microphone not found"**
- ✅ Allow microphone access in your browser
- ✅ Verify that your microphone works in other applications
- ✅ Try different browsers (Chrome recommended)

#### **Error: "Database connection failed"**
- ✅ Verify Supabase credentials in `.env`
- ✅ Make sure tables are created correctly
- ✅ Check internet connectivity

#### **Slow responses or no voice:**
- ✅ Verify your internet connection
- ✅ Check that you have credits in ElevenLabs
- ✅ Try a faster model in OpenRouter

#### **Language Issues:**
- ✅ Miguel speaks Spanish - make sure you're interacting in Spanish
- ✅ The system prompt is optimized for Spanish responses
- ✅ For English adaptation, modify `data/system_prompt.txt`

### **Logs and Debugging:**

The system includes detailed colored logging:
- 🔵 **INFO** - General information
- 🟡 **WARN** - Warnings
- 🔴 **ERROR** - Errors
- 🟢 **SUCCESS** - Successful operations
- 🟣 **DEBUG** - Debug information

---

## 📊 Order Dashboard

Orders can be viewed in real-time through the web dashboard:
🔗 **[Kitchen Dashboard](https://kitchen-dashboard-seven.vercel.app/)**

This dashboard shows:
- 📋 Real-time orders
- ⏰ Timestamps for each order
- 🍽️ Complete details of each order
- 📊 Order status

---

## 🌍 Language Support

### **Current Language:**
- 🇪🇸 **Spanish** - Miguel operates entirely in Spanish with authentic Cadiz expressions
- The voice assistant is optimized for Spanish speakers
- Menu and responses are in Spanish

### **Adapting to Other Languages:**
To adapt WAIter to other languages:

1. **Modify the system prompt** in `data/system_prompt.txt`
2. **Translate the menu** in `data/carta.md`
3. **Adjust voice settings** in ElevenLabs for the target language
4. **Update the personality** to match local culture

---

## 🤝 Contributing

### **How to Contribute:**
1. **Fork** the repository
2. Create a **branch** for your feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. Open a **Pull Request**

### **Areas for Improvement:**
- 🌍 **Translation to other languages**
- 🎨 **User interface improvements**
- 🔧 **New tools** for the agent
- 🍽️ **Menu templates** for different types of restaurants
- 📱 **Native mobile application**
- 🧠 **Conversational model improvements**
- 🗣️ **Multi-language voice support**

---

## 📄 License

This project is under the MIT License. See the `LICENSE` file for more details.

---

## 🙏 Acknowledgments

- **OpenAI** for Whisper and inspiration in conversational models
- **Google** for Gemini 2.5 Flash
- **ElevenLabs** for voice synthesis technology
- **LangChain/LangGraph** for the agent framework
- **Gradio** for the easy-to-use interface
- **HuggingFace** for hosting and embeddings
- **Supabase** for real-time database

---

## 👨‍💻 Author

**Roberto** - Lead Developer
- 🌐 **Web:** [Author information]
- 📧 **Email:** [Author email]
- 🐦 **Twitter:** [Author twitter]

---

## 🔗 Useful Links

- 🚀 **[Live Demo](https://huggingface.co/spaces/ItzRoBeerT/WAIter?logs=container)** - Try WAIter now
- 📊 **[Kitchen Dashboard](https://kitchen-dashboard-seven.vercel.app/)** - View orders in real-time
- 📖 **[LangGraph Documentation](https://github.com/langchain-ai/langgraph)** - Agent framework
- 🔊 **[ElevenLabs API](https://elevenlabs.io/docs)** - Voice synthesis documentation
- 🧠 **[OpenRouter](https://openrouter.ai/)** - AI model access

---

<div align="center">

**⭐ Don't forget to star the project if you found it useful! ⭐**

**🍽️ ¡Que aproveche with Miguel! 🍽️**

</div>