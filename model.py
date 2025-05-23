from os import getenv
from typing import Optional
from langchain_openai import ChatOpenAI

class ModelManager:
    """
    A class to manage the creation and configuration of language model instances.
    Handles API keys and provides fallbacks if environment variables are not found.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "google/gemini-2.5-flash-preview",
        helicone_api_key: Optional[str] = None,
    ):
        """
        Initialize the ModelManager with the specified parameters.
        
        Args:
            api_key: The API key for the language model service. If None, will try to get from environment.
            api_base: The base URL for the language model service. If None, will try to get from environment.
            model_name: The name of the model to use.
            helicone_api_key: The Helicone API key for logging. If None, will try to get from environment.
        """
        
        self.api_key = api_key
        if not self.api_key:
            print("Warning: No API key found.")
            
        self.api_base = api_base 
        if not self.api_base:
            print("Warning: No API base URL found.")
            
        self.helicone_api_key = helicone_api_key
        
        self.model_name = model_name
        
    def create_model(self, **kwargs) -> Optional[ChatOpenAI]:
        """
        Create and return a configured language model instance.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the model constructor.
            
        Returns:
            A configured ChatOpenAI instance, or None if required parameters are missing.
        """
        # Check if required parameters are available
        if not self.api_key or not self.api_base:
            print("Error: Cannot create model. Missing required API key or base URL.")
            return None
            
        # Prepare model kwargs
        model_kwargs = kwargs.get("model_kwargs", {})
        
        # Add Helicone headers if available
        if self.helicone_api_key:
            extra_headers = model_kwargs.get("extra_headers", {})
            extra_headers["Helicone-Auth"] = f"Bearer {self.helicone_api_key}"
            model_kwargs["extra_headers"] = extra_headers
            
        # Update kwargs with new model_kwargs
        kwargs["model_kwargs"] = model_kwargs
            
        # Create and return the model
        try:
            return ChatOpenAI(
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                model_name=self.model_name,
                **kwargs
            )
        except Exception as e:
            print(f"Error creating model: {str(e)}")
            return None