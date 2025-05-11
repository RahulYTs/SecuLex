// Function to copy message content to clipboard
function copyMessageToClipboard(button) {
    // Find the closest message container
    const messageDiv = button.closest('.message');
    
    // Get the message content
    const messageContent = messageDiv.querySelector('.message-content, .content-container');
    
    // Create a temporary div to extract text content without HTML
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = messageContent.innerHTML;
    const textToCopy = tempDiv.textContent || tempDiv.innerText || '';
    
    // Copy to clipboard
    navigator.clipboard.writeText(textToCopy)
        .then(() => {
            // Update button to show success
            button.innerHTML = '<i class="fas fa-check"></i> Copied!';
            button.classList.add('copied');
            
            // Reset after 2 seconds
            setTimeout(() => {
                button.innerHTML = '<i class="fas fa-copy"></i> Copy';
                button.classList.remove('copied');
            }, 2000);
        })
        .catch(err => {
            console.error('Could not copy text: ', err);
            button.innerHTML = '<i class="fas fa-times"></i> Failed';
            
            // Reset after 2 seconds
            setTimeout(() => {
                button.innerHTML = '<i class="fas fa-copy"></i> Copy';
            }, 2000);
        });
}

document.addEventListener('DOMContentLoaded', function() {
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const messageArea = document.getElementById('message-area');
    const clearButton = document.getElementById('clear-history');
    
    // Focus on input field when page loads
    messageInput.focus();
    
    // Scroll to bottom of message area
    function scrollToBottom() {
        messageArea.scrollTop = messageArea.scrollHeight;
    }
    
    // Scroll to bottom initially (in case there are stored messages)
    scrollToBottom();
    
    // Function to add a new message to the UI
    function addMessage(content, role, source = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        
        if (role === 'user') {
            messageDiv.classList.add('user-message');
            
            // For user messages, escape HTML to prevent injection
            // Format code blocks in messages
            let formattedContent = formatCodeBlocks(content);
            
            // Add links to URLs
            formattedContent = linkifyUrls(formattedContent);
            
            messageDiv.innerHTML = formattedContent;
        } else {
            messageDiv.classList.add('assistant-message');
            
            // Create a message header for assistant messages
            const messageHeader = document.createElement('div');
            messageHeader.classList.add('message-header');
            
            // Add a copy to clipboard button
            const copyButton = document.createElement('button');
            copyButton.classList.add('copy-button');
            copyButton.innerHTML = '<i class="fas fa-copy"></i> Copy';
            copyButton.title = 'Copy to clipboard';
            
            copyButton.addEventListener('click', function() {
                // Create a cleaned version of the text without HTML tags
                let textToCopy = content;
                
                // If there's HTML, we need to create a temporary div to extract text
                if (content.includes('<')) {
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = content;
                    textToCopy = tempDiv.textContent || tempDiv.innerText || '';
                }
                
                navigator.clipboard.writeText(textToCopy).then(() => {
                    // Show success message
                    copyButton.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    copyButton.classList.add('copied');
                    
                    // Reset after 2 seconds
                    setTimeout(() => {
                        copyButton.innerHTML = '<i class="fas fa-copy"></i> Copy';
                        copyButton.classList.remove('copied');
                    }, 2000);
                }).catch(err => {
                    console.error('Could not copy text: ', err);
                    copyButton.innerHTML = '<i class="fas fa-times"></i> Failed';
                    
                    // Reset after 2 seconds
                    setTimeout(() => {
                        copyButton.innerHTML = '<i class="fas fa-copy"></i> Copy';
                    }, 2000);
                });
            });
            
            messageHeader.appendChild(copyButton);
            messageDiv.appendChild(messageHeader);
            
            // Create the content container
            const messageContentContainer = document.createElement('div');
            messageContentContainer.classList.add('content-container');
            
            // For assistant messages, we'll support HTML formatting
            // But need to be careful about XSS
            if (content.includes('<p>') || content.includes('<ul>') || content.includes('<ol>') || 
                content.includes('<div') || content.includes('<h3')) {
                
                // Check if this is a structured answer with doc-section
                if (content.includes('doc-section')) {
                    messageDiv.classList.add('structured-response');
                }
                
                // Content already contains HTML formatting, use it directly
                // But still ensure code blocks and links are formatted properly
                messageContentContainer.innerHTML = content
                    // Format any code blocks if they aren't already
                    .replace(/```([\s\S]*?)```/g, function(match, code) {
                        return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
                    })
                    // Format inline code if not already
                    .replace(/`([^`]+)`/g, function(match, code) {
                        return `<code>${escapeHtml(code)}</code>`;
                    });
                
                // Additional safety: Add link attributes to any links
                const links = messageContentContainer.querySelectorAll('a');
                links.forEach(link => {
                    if (!link.hasAttribute('target')) {
                        link.setAttribute('target', '_blank');
                        link.setAttribute('rel', 'noopener noreferrer');
                    }
                });
            } else {
                // No HTML formatting, apply our regular formatting
                let formattedContent = formatCodeBlocks(content);
                formattedContent = linkifyUrls(formattedContent);
                messageContentContainer.innerHTML = formattedContent;
            }
            
            messageDiv.appendChild(messageContentContainer);
        }
        
        // Add source tag if provided (for assistant messages)
        if (source && role === 'assistant') {
            const footer = document.createElement('div');
            footer.classList.add('message-footer');
            
            let sourceIcon = 'fa-globe';
            let sourceText = source;
            
            if (source.includes('database')) {
                sourceIcon = 'fa-database';
                sourceText = 'From knowledge base';
            } else if (source.includes('web')) {
                sourceText = 'From web search';
            }
            
            footer.innerHTML = `<span class="source-tag"><i class="fas ${sourceIcon} me-1"></i>${sourceText}</span>`;
            
            messageDiv.appendChild(footer);
        }
        
        messageArea.appendChild(messageDiv);
        scrollToBottom();
    }
    
    // Format code blocks in messages - detects ```code``` syntax
    function formatCodeBlocks(text) {
        // Replace markdown-style code blocks with HTML
        return text.replace(/```([\s\S]*?)```/g, function(match, code) {
            return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
        }).replace(/`([^`]+)`/g, function(match, code) {
            return `<code>${escapeHtml(code)}</code>`;
        });
    }
    
    // Escape HTML special characters
    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Convert URLs to clickable links
    function linkifyUrls(text) {
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        return text.replace(urlRegex, function(url) {
            return `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`;
        });
    }
    
    // Add loading indicator
    function addLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'assistant-message', 'loading-message');
        loadingDiv.innerHTML = '<div class="loading-indicator"></div> Thinking...';
        messageArea.appendChild(loadingDiv);
        scrollToBottom();
        return loadingDiv;
    }
    
    // Remove loading indicator
    function removeLoadingIndicator(indicator) {
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
    }
    
    // Handle form submission
    messageForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const query = messageInput.value.trim();
        if (!query) return;
        
        // Add user message to UI
        addMessage(query, 'user');
        
        // Clear input
        messageInput.value = '';
        
        // Add loading indicator
        const loadingIndicator = addLoadingIndicator();
        
        // Set a timeout to detect long-running requests
        const timeoutId = setTimeout(() => {
            console.log("Request is taking longer than expected...");
            // Update the loading message if it's still visible
            if (loadingIndicator && loadingIndicator.parentNode) {
                loadingIndicator.innerHTML = '<div class="loading-indicator"></div> Searching the web, this might take a moment...';
            }
        }, 5000);
        
        // Send query to server
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query }),
            // Add a longer timeout for the fetch request
            timeout: 30000
        })
        .then(response => {
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                if (response.status === 500) {
                    throw new Error('Server error. The system might be experiencing issues.');
                } else if (response.status === 404) {
                    throw new Error('API endpoint not found.');
                } else if (response.status === 0) {
                    throw new Error('Network connection lost. Please check your internet connection.');
                } else {
                    throw new Error(`Server responded with status code: ${response.status}`);
                }
            }
            return response.json();
        })
        .then(data => {
            // Remove loading indicator
            removeLoadingIndicator(loadingIndicator);
            
            if (data.error) {
                addMessage(`Error: ${data.error}`, 'assistant');
            } else {
                // Add assistant response to UI with source information
                let sourceInfo = data.source;
                
                // Add additional metadata if available
                if (data.metadata) {
                    if (data.source === 'database' && data.metadata.confidence) {
                        sourceInfo += ` (confidence: ${data.metadata.confidence})`;
                    }
                }
                
                addMessage(data.response, 'assistant', sourceInfo);
                
                // Log if this was from database or web search
                console.log(`Response source: ${data.source}`);
                if (data.metadata) {
                    console.log('Response metadata:', data.metadata);
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
            clearTimeout(timeoutId);
            removeLoadingIndicator(loadingIndicator);
            
            // Create a more user-friendly error message
            let errorMessage = "I'm sorry, I encountered an error while processing your request.";
            
            if (error.message.includes('internet') || error.message.includes('network') || 
                error.message.includes('connection')) {
                errorMessage += " It looks like there might be a network connectivity issue.";
            } else if (error.message.includes('timeout')) {
                errorMessage += " The request timed out, which could be due to slow internet or the server being busy.";
            }
            
            errorMessage += " Please try again in a moment.";
            
            addMessage(errorMessage, 'assistant');
        });
    });
    
    // Handle clear history button
    clearButton.addEventListener('click', function() {
        // Confirm before clearing
        if (confirm('Are you sure you want to clear the chat history?')) {
            // Clear UI
            messageArea.innerHTML = '';
            
            // Send request to clear history on server
            fetch('/clear_history', {
                method: 'POST'
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to clear history on server');
                }
                return response.json();
            })
            .then(data => {
                console.log('History cleared');
            })
            .catch(error => {
                console.error('Error clearing history:', error);
            });
        }
    });
    
    // Handle Enter key for sending message (Shift+Enter for new line)
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            messageForm.dispatchEvent(new Event('submit'));
        }
    });
});
