package com.library.controller.web;

import com.library.entity.Book;
import com.library.service.BookService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
@RequestMapping("/books")
public class BookWebController {
    private final BookService bookService;

    @Autowired
    public BookWebController(BookService bookService) {
        this.bookService = bookService;
    }

    @GetMapping
    public String listBooks(Model model) {
        model.addAttribute("books", bookService.getAllBooks());
        return "books/list";
    }

    @GetMapping("/create")
    public String showCreateForm(Model model) {
        System.out.println("Accessing create book form");
        if (!model.containsAttribute("book")) {
            model.addAttribute("book", new Book());
        }
        return "books/create";
    }

    @PostMapping("/create")
    public String createBook(@Valid @ModelAttribute Book book, BindingResult result, Model model) {
        System.out.println("Attempting to create book: " + book.getTitle());
        
        if (result.hasErrors()) {
            System.out.println("Validation errors: " + result.getAllErrors());
            return "books/create"; // Return to the form if there are validation errors
        }
        
        try {
            bookService.saveBook(book);
            System.out.println("Book saved successfully: " + book.getTitle());
        } catch (Exception e) {
            System.err.println("Error saving book: " + e.getMessage());
            model.addAttribute("error", "An error occurred while saving the book.");
            return "books/create"; // Return to the form with an error message
        }
        
        return "redirect:/books"; // Redirect to the list of books after saving
    }

    @GetMapping("/edit/{id}")
    public String showEditForm(@PathVariable Long id, Model model) {
        bookService.getBookById(id).ifPresent(book -> model.addAttribute("book", book));
        return "books/edit";
    }

    @PostMapping("/edit/{id}")
    public String updateBook(@PathVariable Long id, @Valid @ModelAttribute Book book, BindingResult result) {
        if (result.hasErrors()) {
            return "books/edit";
        }
        bookService.updateBook(id, book);
        return "redirect:/books";
    }

    @GetMapping("/delete/{id}")
    public String deleteBook(@PathVariable Long id) {
        bookService.deleteBook(id);
        return "redirect:/books";
    }

    @GetMapping("/search")
    public String searchBooks(@RequestParam(required = false) String keyword, Model model) {
        model.addAttribute("books", 
            keyword != null ? bookService.searchBooks(keyword) : bookService.getAllBooks());
        return "books/list";
    }

    @GetMapping("/category/{category}")
    public String getBooksByCategory(@PathVariable String category, Model model) {
        model.addAttribute("books", bookService.getBooksByCategory(category));
        model.addAttribute("currentCategory", category);
        return "books/list";
    }
} 