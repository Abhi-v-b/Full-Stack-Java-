package com.library.controller.web;

import com.library.service.BookService;
import com.library.service.UserService;
import com.library.service.BorrowingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
@RequestMapping("/borrowings")
public class WebBorrowingController {

    private final BorrowingService borrowingService;
    private final BookService bookService;
    private final UserService userService;

    @Autowired
    public WebBorrowingController(BorrowingService borrowingService, 
                                BookService bookService, 
                                UserService userService) {
        this.borrowingService = borrowingService;
        this.bookService = bookService;
        this.userService = userService;
    }

    @GetMapping
    public String listBorrowings(@RequestParam(required = false) String status, Model model) {
        if (status != null) {
            model.addAttribute("borrowings", borrowingService.getCurrentBorrowings());
        } else {
            model.addAttribute("borrowings", borrowingService.getAllBorrowings());
        }
        model.addAttribute("status", status);
        return "borrowings/list";
    }

    @GetMapping("/create")
    public String createBorrowingForm(Model model) {
        model.addAttribute("users", userService.getActiveUsers());
        model.addAttribute("availableBooks", bookService.getAllBooks().stream()
                .filter(book -> book.getAvailableQuantity() > 0)
                .toList());
        return "borrowings/form";
    }

    @PostMapping("/borrow")
    public String createBorrowing(@RequestParam Long userId,
                                @RequestParam Long bookId,
                                @RequestParam(defaultValue = "14") Integer durationInDays,
                                RedirectAttributes redirectAttributes) {
        try {
            borrowingService.borrowBook(userId, bookId, durationInDays);
            redirectAttributes.addFlashAttribute("successMessage", "Book borrowed successfully");
        } catch (RuntimeException e) {
            redirectAttributes.addFlashAttribute("errorMessage", e.getMessage());
        }
        return "redirect:/borrowings";
    }

    @GetMapping("/{id}/return")
    public String returnBook(@PathVariable Long id, RedirectAttributes redirectAttributes) {
        try {
            borrowingService.returnBook(id);
            redirectAttributes.addFlashAttribute("successMessage", "Book returned successfully");
        } catch (RuntimeException e) {
            redirectAttributes.addFlashAttribute("errorMessage", e.getMessage());
        }
        return "redirect:/borrowings";
    }

    @GetMapping("/{id}")
    public String viewBorrowing(@PathVariable Long id, Model model) {
        return borrowingService.getBorrowingById(id)
                .map(borrowing -> {
                    model.addAttribute("borrowing", borrowing);
                    return "borrowings/view";
                })
                .orElse("redirect:/borrowings");
    }

    @GetMapping("/overdue")
    public String listOverdueBorrowings(Model model) {
        model.addAttribute("borrowings", borrowingService.getOverdueBorrowings());
        model.addAttribute("status", "OVERDUE");
        return "borrowings/list";
    }
} 